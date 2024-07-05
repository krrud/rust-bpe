# Byte-Pair Encoding Tokenizer
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)


## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Acknowledgements](#acknowledgements)


## Overview <a name="overview"></a>
This project introduces a robust and lightweight byte-pair encoding (BPE) tokenizer, designed to handle linguistic data efficiently. While it primarily focuses on the English language, its architecture allows it to model any language effectively. The tokenizer is equipped with a comprehensive vocabulary of over 20,000 tokens, demonstrating strong adaptability to a variety of unseen text inputs.

The tokenizer is compiled for web use, leveraging the power of WebAssembly through wasm-bindgen. This makes it readily accessible for integration into web applications, providing a seamless user experience without the need for server-side processing.

**Planned future enhancements include:**
- *GPU Implementation:* Using WGPU for processing larger training corpora and achieving faster training and tokenization.
- *Expanded Language Support:* Developing capabilities for capitalized and multilingual tokenization post-performance improvements.


## Installation <a name="installation"></a>
**To run this project, execute the following commands in your terminal:**
```sh
git clone https://github.com/krrud/rust-bpe.git
cd rust-bpe
cargo build
```


## Usage <a name="usage"></a>
**To train the tokenizer with your corpus:**
```rust
  // Load the dataset from the specified directory
  let corpus = Tokenizer::process_dataset("path/to/dataset");

  // Set the number of iterations for training
  let iterations = 20000;

  // Specify the output path for the trained tokenizer model
  let output = "./src/tokenizer_train.json";

  // Optional: start from a pretrained model if available
  // Replace with Some("path/to/pretrained_tokenizer.json") if applicable
  let pretrained_model = None; 

  // Train the tokenizer
  let tokenizer = Tokenizer::train_cpu(
      &corpus,
      iterations,
      output,
      pretrained_model
  );
```


**To tokenize text to indices, convert to token strings, or detokenize back to the input:**
```rust
let tokenizer = Tokenizer::load("path/to/your/trained_tokenizer.json").unwrap();
let tokens = tokenizer.tokenize("text to tokenize");  // Returns token indices
let token_strings = tokenizer.get_tokens(&tokens);    // Converts indices to associated strings
let detokenized = tokenizer.detokenize(&tokens);      // Converts back to the original text
```


**To use the tokenizer in the browser via Wasm:**
```javascript
async function wasmTokenizer(textInput) {
  try {
    // Import and initialize the module
    const {default: init, TokenizerJS} = await import('/path/to/wasm-pkg');
    await init();

    // Load the vocabulary and merge rules
    const fetchTokenizer = await fetch("/path/to/trained_model.json");
    if (!fetchTokenizer.ok) {
      throw new Error("Failed to fetch tokenizer");
    }
    const {vocabulary, merge_rules} = await fetchTokenizer.json();

    // Instantiate the tokenizer
    const tokenizer = new TokenizerJS(vocabulary, merge_rules);

    // Use the tokenizer
    const indices = tokenizer.tokenize(textInput);
    const strings = tokenizer.getTokens(indices);
    const rebuilt = tokenizer.detokenize(indices);

  } catch (error) {
    console.error("Failed to load WASM module or tokenizer:", error);
  }
};
```


## Acknowledgements <a name="acknowledgements"></a>
Training data was graciously provided by:
- [The Gutenberg Project](https://www.gutenberg.org)
- [Wiki Dumps](https://en.wikipedia.org/wiki/Wikipedia:Database_download)
- [CNN Dailymail Database](https://huggingface.co/datasets/ccdv/cnn_dailymail)


## License <a name="license"></a>
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
