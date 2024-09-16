from train_model import train_model

train_model(
    data_path = "../glue/CoLA/train.json",
    output_dir = "../output/model_for_CoLA",
    prompt_template_path = "../glue/CoLA/template_CoLA.json",
    val_set_size = 0.95,
)
