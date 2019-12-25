# Rinav R Kasthuri

from coremltools import converters

def getCoreML(modelName, authorName, description, inputDescription, outputDescription):
    outputLabels = [str(i) for i in range(41)]
    
    model = converters.keras.convert(modelName + ".h5", input_names = ["image"],
                                                 output_names = ["output"],
                                                 class_labels = outputLabels,
                                                 image_input_names = "image")
    model.author = authorName
    model.short_description = description
    model.input_description["image"] = inputDescription
    model.output_description["output"] = outputDescription
    
    model.save(modelName + ".mlmodel")
    
if __name__ == "__main__":
    getCoreML("Third", "Rinav R Kasthuri", "Crop diseases and Weed detection", 
              "2-D Picture of relevant crop / plant (which must be resized to 256x256)",
              "Prediction of which crop disease / weed")
