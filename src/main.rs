mod tokenizer;
use tokenizer::Tokenizer;


fn main() {
    let source = Tokenizer::process_dataset("F:/datasets/train");
    let iterations = 20000;
    let output = "./src/tokenizer_training.json";
    let tokenizer = Tokenizer::train_cpu(&source, iterations, output, None);
}
