mod tokenizer;
use tokenizer::Tokenizer;


fn main() {
    let source = Tokenizer::process_dataset("F:/datasets/test");
    let iterations = 25000;
    let output = "./src/tokenizer_training.json";
    let tokenizer = Tokenizer::train_cpu(&source, iterations, Some(output));
}
