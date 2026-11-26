#include "helper/resource_path.h"

#include <cstdlib>
#include <filesystem>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace
{

std::string readEnvVar(const char *env_var_name)
{
    if (env_var_name == nullptr || env_var_name[0] == '\0')
    {
        return "";
    }
    const char *raw = std::getenv(env_var_name);
    return (raw != nullptr) ? std::string(raw) : std::string{};
}

bool matchesPathKind(const std::filesystem::path &candidate, bool expect_directory)
{
    std::error_code ec;
    if (expect_directory)
    {
        return std::filesystem::is_directory(candidate, ec) && !ec;
    }
    return std::filesystem::is_regular_file(candidate, ec) && !ec;
}

std::string tryResolveUpwards(const std::filesystem::path &base_dir,
                                const std::filesystem::path &relative_target,
                                bool expect_directory)
{
    if (base_dir.empty() || relative_target.empty() || !relative_target.is_relative())
    {
        return "";
    }

    std::filesystem::path cursor = base_dir.lexically_normal();
    for (int depth = 0; depth <= 4; ++depth)
    {
        const std::filesystem::path probe = (cursor / relative_target).lexically_normal();
        if (matchesPathKind(probe, expect_directory))
        {
            return probe.generic_string();
        }
        if (!cursor.has_parent_path())
        {
            break;
        }
        const std::filesystem::path parent = cursor.parent_path();
        if (parent == cursor)
        {
            break;
        }
        cursor = parent;
    }
    return "";
}

std::string resolveExistingPath(const std::string &requested,
                                const char *env_var_name,
                                const std::string &relative_fallback,
                                bool expect_directory)
{
    namespace fs = std::filesystem;

    const bool use_default_overrides = requested.empty() || (!relative_fallback.empty() && requested == relative_fallback);

    if (!requested.empty())
    {
        fs::path direct(requested);
        if (matchesPathKind(direct, expect_directory))
        {
            return direct.lexically_normal().generic_string();
        }
    }

    if (use_default_overrides)
    {
        const std::string env_override = readEnvVar(env_var_name);
        if (!env_override.empty())
        {
            fs::path env_path(env_override);
            if (matchesPathKind(env_path, expect_directory))
            {
                return env_path.lexically_normal().generic_string();
            }
        }
    }

    const std::string primary = !requested.empty() ? requested : relative_fallback;
    const std::string root_env = readEnvVar("VULKANML_ROOT");
    if (!root_env.empty())
    {
        const fs::path root_path(root_env);
        if (!primary.empty())
        {
            const fs::path probe = (root_path / fs::path(primary)).lexically_normal();
            if (matchesPathKind(probe, expect_directory))
            {
                return probe.generic_string();
            }
        }
    }

    std::error_code cwd_ec;
    const fs::path cwd = fs::current_path(cwd_ec);
    const fs::path exe_dir(Resource_Path::getExecutableDirectory());

    auto search_candidate = [&](const std::string &candidate_str) -> std::string
    {
        if (candidate_str.empty())
        {
            return "";
        }
        const fs::path candidate_path(candidate_str);
        if (!candidate_path.is_relative())
        {
            return "";
        }
        if (!cwd_ec && !cwd.empty())
        {
            if (std::string hit = tryResolveUpwards(cwd, candidate_path, expect_directory); !hit.empty())
            {
                return hit;
            }
        }
        if (!exe_dir.empty())
        {
            if (std::string hit = tryResolveUpwards(exe_dir, candidate_path, expect_directory); !hit.empty())
            {
                return hit;
            }
        }
        return "";
    };

    if (std::string hit = search_candidate(primary); !hit.empty())
    {
        return hit;
    }

    if (!primary.empty())
    {
        return fs::path(primary).lexically_normal().generic_string();
    }
    return "";
}

std::string findProjectRootFromBase(const std::filesystem::path &base_dir)
{
    if (base_dir.empty())
    {
        return "";
    }
    std::filesystem::path cursor = base_dir.lexically_normal();
    for (int depth = 0; depth <= 4; ++depth)
    {
        std::error_code ec1;
        std::error_code ec2;
        if (std::filesystem::is_directory(cursor / "compute_shader", ec1) && !ec1 &&
            std::filesystem::is_regular_file(cursor / "compute_shader" / "shader_dictionary.json", ec2) && !ec2)
        {
            return cursor.generic_string();
        }
        if (!cursor.has_parent_path())
        {
            break;
        }
        const std::filesystem::path parent = cursor.parent_path();
        if (parent == cursor)
        {
            break;
        }
        cursor = parent;
    }
    return "";
}

}

namespace Resource_Path
{

std::string getExecutableDirectory()
{
    namespace fs = std::filesystem;
#ifdef _WIN32
    std::vector<wchar_t> buffer(1024, L'\0');
    for (;;)
    {
        DWORD copied = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (copied == 0)
        {
            return "";
        }
        if (copied < buffer.size() - 1)
        {
            fs::path exe_path(std::wstring(buffer.data(), copied));
            return exe_path.parent_path().lexically_normal().generic_string();
        }
        if (buffer.size() >= 32768)
        {
            return "";
        }
        buffer.resize(buffer.size() * 2, L'\0');
    }
#else
    std::vector<char> buffer(1024, '\0');
    ssize_t len = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    if (len <= 0)
    {
        return "";
    }
    fs::path exe_path(std::string(buffer.data(), static_cast<size_t>(len)));
    return exe_path.parent_path().lexically_normal().generic_string();
#endif
}

std::string resolveDirectory(const std::string &requested, const char *env_var_name, const std::string &relative_fallback)
{
    return resolveExistingPath(requested, env_var_name, relative_fallback, true);
}

std::string resolveFile(const std::string &requested, const char *env_var_name, const std::string &relative_fallback)
{
    return resolveExistingPath(requested, env_var_name, relative_fallback, false);
}

std::string resolveWritableDirectory(const std::string &requested, const char *env_var_name, const std::string &relative_fallback)
{
    namespace fs = std::filesystem;

    const std::string env_override = readEnvVar(env_var_name);
    if (!env_override.empty())
    {
        return fs::path(env_override).lexically_normal().generic_string();
    }

    const std::string candidate = !requested.empty() ? requested : relative_fallback;
    if (candidate.empty())
    {
        return ".";
    }

    const fs::path candidate_path(candidate);
    if (!candidate_path.is_relative() || matchesPathKind(candidate_path, true))
    {
        return candidate_path.lexically_normal().generic_string();
    }

    const std::string existing = resolveExistingPath(candidate, nullptr, relative_fallback, true);
    if (!existing.empty() && matchesPathKind(fs::path(existing), true))
    {
        return existing;
    }

    if (const std::string root_env = readEnvVar("VULKANML_ROOT"); !root_env.empty())
    {
        return (fs::path(root_env) / candidate_path).lexically_normal().generic_string();
    }

    std::error_code cwd_ec;
    const fs::path cwd = fs::current_path(cwd_ec);
    if (!cwd_ec && !cwd.empty())
    {
        if (std::string project_root = findProjectRootFromBase(cwd); !project_root.empty())
        {
            return (fs::path(project_root) / candidate_path).lexically_normal().generic_string();
        }
    }

    const fs::path exe_dir(getExecutableDirectory());
    if (!exe_dir.empty())
    {
        if (std::string project_root = findProjectRootFromBase(exe_dir); !project_root.empty())
        {
            return (fs::path(project_root) / candidate_path).lexically_normal().generic_string();
        }
    }

    return candidate_path.lexically_normal().generic_string();
}

}
