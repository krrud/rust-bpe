mod tokenizer;
use tokenizer::Tokenizer;


fn main() {
    let source = Tokenizer::process_dataset("F:/datasets/test");
    let iterations = 10000;
    let output = "./src/tokenizer_train.json";
    let pretrained_model_path = None;
    let tokenizer = Tokenizer::train_cpu(&source, iterations, output, pretrained_model_path);
}
