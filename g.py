from model_inference_v1 import generate
import fire

def generate_glue(task_type: str = "CoLA"):
    generate(
        data_path = "../glue/"+task_type+"/train.json",
        val_set_size=16,
        task_type=task_type,
        data_catagory="train",
        generate_train_or_test="test",
        output_file="../output/otuput_"+task_type+".txt",
        prompt_template_path="../glue/"+task_type+"/template_"+task_type+".json",
    )
if __name__ == '__main__':
    fire.Fire(generate_glue)