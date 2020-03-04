#!/usr/bin/env python

fix parameter
for name, param in model.named_parameters():
    if name.startswith('embeddings'):
        param.requires_grad = False
if freeze_embeddings:
    for param in list(model.bert.embeddings.parameters()):
        param.requires_grad = False
    print("Froze Embedding Layer")

freeze_layers is a string "1,2,3" representing layer number
freeze_layers = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"
if freeze_layers is not "":
    layer_indexes = [int(x) for x in freeze_layers.split(",")]
    for layer_idx in layer_indexes:
        for param in list(model.bert.encoder.layer[layer_idx].parameters()):
            param.requires_grad = False
        print("Froze Layer: ", layer_idx)
