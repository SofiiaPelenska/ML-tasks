import ipywidgets as widgets
from IPython.display import display
from PIL import Image
import tempfile
from src.utils.labels import CLASS_MAP


def interactive_demo(pipeline):
    POSSIBLE_CLASSES = list(CLASS_MAP.values())

    classes_label = widgets.HTML(
        value="<b>Possible classes:</b> " + ", ".join(POSSIBLE_CLASSES)
    )

    text_input = widgets.Text(
        value="There is a cow in the picture",
        description="Sentence:",
        layout=widgets.Layout(width='500px')
    )

    upload = widgets.FileUpload(
        accept="image/*",
        multiple=False,
        description="Upload image"
    )

    output = widgets.Output()

    def run_pipeline(button):
        with output:
            output.clear_output()

            if len(upload.value) == 0:
                print("Please upload an image")
                return

            file_info = upload.value[0]
            image_bytes = file_info["content"]

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            temp_file.write(image_bytes)
            temp_file.close()

            text = text_input.value

            result = pipeline.run(text, temp_file.name)

            img = Image.open(temp_file.name)
            img.thumbnail((300, 300))
            display(img)

            print("\nText:", text)
            print("Animals in text:", result["animals_in_text"])
            print("Image prediction:", result["image_prediction"])
            print("Match:", result["result"])

    run_button = widgets.Button(description="Run pipeline")

    run_button.on_click(run_pipeline)

    display(classes_label) 
    display(text_input)
    display(upload)
    display(run_button)
    display(output)