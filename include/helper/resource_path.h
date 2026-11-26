#pragma once

#include <string>

namespace Resource_Path
{

std::string getExecutableDirectory();
std::string resolveDirectory(const std::string &requested, const char *env_var_name = nullptr, const std::string &relative_fallback = "");
std::string resolveFile(const std::string &requested, const char *env_var_name = nullptr, const std::string &relative_fallback = "");
std::string resolveWritableDirectory(const std::string &requested, const char *env_var_name = nullptr, const std::string &relative_fallback = "");

}
