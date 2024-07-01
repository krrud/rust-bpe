mod gpu;
mod tokenizer;
use tokenizer::Tokenizer;

#[tokio::main]
async fn main() {
    let source = Tokenizer::process_dataset("F:/datasets/gutenberg");
    let iterations = 2;
    let output = "./src/tokenizer_new.json";
    let tokenizer = Tokenizer::train_cpu(&source, iterations, Some(output));
    tokenizer.save(output).unwrap();
}

