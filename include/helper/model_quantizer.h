#pragma once

#include "layer/ilayer.h"
#include "neural_network.h"
#include "llm/causal_lm.h"

class Model_Quantizer
{
public:
    static void quantizeLayer(ILayer &layer)
    {
        layer.quantizeToTernary();
    }

    static void quantizeToTernary(Neural_Network &net)
    {
        net.enableTernary(true);
        net.setTrainingMode(false);
        net.prepareInference();
    }

    static void quantizeToTernary(Causal_LM &model)
    {
        model.quantizeToTernary();
    }
};
