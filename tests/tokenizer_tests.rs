use rust_bpe::tokenizer::Tokenizer;


#[test]
fn train_tokenizer() {
    let source = Tokenizer::process_dataset("F:/datasets/tokenizer/train");
    let iterations = 25000;
    let output = "./src/models/train.json";
    let pretrained_model = None;
    let tokenizer = Tokenizer::train_cpu(&source, iterations, output, pretrained_model);
}

#[test]
fn validate_tokenizer() {
    let text = Tokenizer::process_dataset("F:/datasets/tokenizer/validate/");
    let tokenizer = Tokenizer::load("./src/models/rust-bpe-uncased-25k.json").unwrap();

    let time = std::time::Instant::now();
    let tokens = tokenizer.tokenize(&text);
    println!("Tokenization time: {:?}", time.elapsed());
    let token_vals = tokenizer.get_tokens(&tokens);
    let token_indices = tokenizer.get_indices(&token_vals);
    let detokenized = tokenizer.detokenize(&tokens);

    println!("Input: {:?}", &text);
    println!("Tokens: {:?}", token_vals);
    println!("Detokenized: {:?}", detokenized);
    println!("Num tokens: {:?}", tokens.len());
    println!("Vocabulary size: {:?}", tokenizer.vocabulary.len());

    let expected = Tokenizer::clean_text(&text);
    let actual = detokenized;
    assert_eq!(actual.trim(), expected.trim());
}

