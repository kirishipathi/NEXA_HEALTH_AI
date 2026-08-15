# EndoNexa AI Notes

## Known Issues

The live prediction pipeline currently shows reduced sensitivity to positive (endometriosis-suspected) cases in manual testing, despite Phase 4 offline evaluation reporting 70% recall. Root cause under investigation — suspected label-orientation mismatch between the image and symptom branches. This does not affect the validity of the reported Phase 3/4 training and evaluation metrics, which were computed correctly on the test set.

## Grad-CAM limitation

The image prediction pipeline uses an EfficientNetB0 model exported from the training pipeline. During runtime validation, Grad-CAM generation hit a real Keras graph-connectivity limitation when the exported model was saved as a nested submodel inside the outer wrapper model.

The observed runtime error was:

`ValueError: Output with path 0 is not connected to inputs`

This means the image model can still be used for prediction, but a connected Grad-CAM gradient graph cannot be built from the nested export in its current form. The current project therefore treats Grad-CAM as optional: predictions remain valid even when the overlay is unavailable.

A future fix would involve re-exporting the image model as a flat functional model, with the target feature layer directly connected to the true model input tensor, so that a valid Grad-CAM `grad_model` can be constructed and traced during inference.
