# Byte-Pair Encoding Tokenizer
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)


## Table of Contents
- [Overview](#overview)
- [Technology Stack](#stack)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Acknowledgements](#acknowledgements)


## Overview <a name="overview"></a>

This project introduces a robust and efficient tokenizer using Byte-Pair Encoding (BPE) technology, tailored for processing and analyzing linguistic data. Initially configured for the English language, the tokenizer's flexible architecture is capable of supporting a wide range of languages, making it an ideal solution for global text processing applications.

- **Key Features:** Comes pre-loaded with a rich set of over 25,000 tokens, allowing for effective handling and adaptability to diverse and unseen textual inputs.

- **High Performance:** Optimized for speed and efficiency, the tokenizer processes large volumes of text swiftly, facilitating rapid data throughput without sacrificing accuracy.

- **Multi-language Capability:** Designed to be language-agnostic, offering potential customization options for modeling specific linguistic characteristics of any target language.


## Technology Stack <a name="stack"></a>

- **Rust:** Employs Rust’s powerful performance characteristics and memory safety guarantees to ensure high-speed data processing with minimal overhead. Rust's robust type system and ownership model eliminate common bugs, making it ideal for building reliable and efficient tokenization tools.

- **PyO3:** Enables seamless integration with Python, allowing the tokenizer to be used as a native Python extension. This integration provides the benefits of Rust's performance and safety in Python's flexible and dynamic ecosystem.

- **Web Assembly:** Compiled to WebAssembly for high-performance use in web applications, ensuring that the tokenizer can be run directly in the browser with near-native speed.

For detailed instructions on how to integrate and utilize the tokenizer in both Python environments and web applications, please refer to the [Usage](#usage) section.

**Planned future enhancements include:**
- **GPU Implementation:** Using WGPU for processing larger training corpora and achieving faster training and tokenization.
- **Expanded Language Support:** Developing capabilities for capitalized and multilingual tokenization post-performance improvements.


## Installation <a name="installation"></a>

**To run this project, execute the following commands in your terminal:**
```sh
git clone https://github.com/krrud/rust-bpe.git
cd rust-bpe
cargo build
```


## Usage <a name="usage"></a>

**To use the tokenizer in python:**

*First install the .whl:*
```sh
pip install path/to/tokenizer-build.whl
```

*Then use it as follows:*
```python
import rust_bpe

tokenizer = rust_bpe.TokenizerPy("/path/to/trained-tokenizer.json")
tokenized = tokenizer.tokenize("some text to tokenize")
detokenized = tokenizer.detokenize(tokenized)
```


**To use the tokenizer in the browser via Wasm:**
```javascript
async function wasmTokenizer(textInput) {
  try {
    // Import and initialize the module
    const {default: init, TokenizerJs} = await import('/path/to/wasm-pkg');
    await init();

    // Load the trained tokenizer
    const fetchTokenizer = await fetch("/path/to/trained-tokenizer.json");
    if (!fetchTokenizer.ok) {
      throw new Error("Failed to fetch tokenizer");
    }
    const {vocabulary, merge_rules, config} = await fetchTokenizer.json();

    // Instantiate the tokenizer
    const tokenizer = new TokenizerJs(vocabulary, merge_rules, config);

    // Use the tokenizer
    const tokenized = tokenizer.tokenize(textInput);
    const detokenized = tokenizer.detokenize(indices);

  } catch (error) {
    console.error("Failed to load WASM module or tokenizer:", error);
  }
};
```

**To train the tokenizer with Rust directly:**
```rust
  // Load the dataset from the specified directory
  let corpus = Tokenizer::process_dataset("path/to/dataset");

  // Set the number of iterations for training
  let iterations = 25000;

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


## Acknowledgements <a name="acknowledgements"></a>
Training data was graciously provided by:
- [The Gutenberg Project](https://www.gutenberg.org)
- [Wiki Dumps](https://en.wikipedia.org/wiki/Wikipedia:Database_download)
- [CNN Dailymail Database](https://huggingface.co/datasets/ccdv/cnn_dailymail)


## License <a name="license"></a>
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
