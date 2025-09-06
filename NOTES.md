# Notes

## Python:

* Raw strings i.e., `r""` are treated as-is. In other words, escape sequences like `\n` or `\t` will not be expanded.
  The only exception is a backslash, `\`, which still needs to be backslashed to be interprted correctly i.e., `\\`.
* When using regular expression for splitting, the `()` can be used to capture the separator itself.
* In `re.split(r'([\.,:;\?_!]|--|\s)', "Hello, World. I am alive?")`, all separators are enclosed within `()`. Also,
  multiple separators are separated using `|`. Punctuations are grouped inside `[]` that signifies on one of.  Backslash
  befor `?` and `.` are possibly not required, but it doesn't hurts.
* In `re.sub(r'\s+([,.?!"()\'])', r'\1', text)`, we are essentially replacing the occurence ` ;` with `;`. The `\1` is a
  reference to the first capture group.  The replacement is necessary because in its absence we will get teh text `Hello
  World !` as opposed to `Hello World!`.


## Deep Learning

* Sigmoid is represented as $`f(x)=1/(1+\exp(-x))`$, which is the same as $`f(x)=\exp(x)/(1+\exp(x))`$. As `x`
  approaches infinity, `f(x)` approaches 1. `1` can be replaced by any other positive value, say `L`, which then becomes
  the upper limit. The range of the sigmoid function is `[0, 1]` or `[0, L]`, and the midpoint is `0.5`.
* ReLU, rectified linear unit, is represented as `max(0, x)`.
* The function `binary_cross_entropy` returns a scalar.  It is used to compare `input` probabilities (in the range [0,
  1]) to `target` class labels (either 0 or 1).
* A linear regression is represented by $`y = w*x + b`$, where `w` is weight and `b` is bias. The snippet below shows
  some examples, ``` w*x -> [128, 784] * [784, 1] -> [128, 1] w*x -> [64, 128] * [128, 1] -> [64, 1] ```
  * The thing to keep in mind is that we are multiplying `w` with the incoming vector `x`, not the other way around.
  * The first dimension of the weight vector represents the number of nodes in the next layer of the neural network.
  * The second dimension of the weight vector represents the dimension of the input vector.
* `Dataset` and `DataLoader` classes allows for encapsulating and loading the dataset in question. The `Dataset` is an
  interface class, and needs to be extended with the `__getitem__` and `__len__` methods. The `DataLoader` class takes
  in the `Dataset` instance and allows for specifying options such as,
  * Batch size
  * Shuffle
  * Drop the last set of items if the dataset length is not a multiple of the batch size.
* Recurrent Neural Networks (RNNs)
  * Used on sequential or time series data where order matters
  * The "memory" from earlier inputs is used to process the current input. The "memory" is essentially a hidden state
    that is transferred from the last input to the next.
  * Encoder-decoder RNNs:
    * Both the encoder and the decoder are layers of RNNs stacked together. Each layer uses the input at the current
      timestep and the last "hidden" state (essentially a vector) to compute the updated "hidden" state.
    * For eg., in the text sequence "Roses are red", the ordering of the individual words provide the relevant context.
      The sequence is possibly tokenized, and fed one token at a time into the encoder.
    * The overall "hidden" vector state is only ready once the entire text sequence is processed.
    * One issue with the encoder-decoder RNNs is the inability of the decoder to access any specific position in the
      input sequence. All there is, is a "hidden" vector state that can be accessed.
    * The "hidden" vector state is also referred to as "memory".
    * Open question: What determines the number of layers in the encoder or the decoder?
    * Resources:
      * https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks
      * https://www.ibm.com/think/topics/recurrent-neural-networks
      * https://medium.com/analytics-vidhya/encoders-decoders-sequence-to-sequence-architecture-5644efbb3392
  * Self attention:
    * Decoder only transformer
    * Attention weights
      * For a given input sequence, the attention weights for a specific token involves computing the similarity with
        all other tokens. The similarity is computed as dot product of token embeddings, i.e., for token 1, its
        attention weight is computed as the following tensor, ``` attention_weight_token_1 = [dot(token_e_0, token_e_1),
        dot(token_e_1, token_e_1), dot(token_e_2, token_e_1), ..., dot(token_e_n, token_1)] ```
      * The tensor is further normalized, i.e., ``` softmax(attention_weight_token_1) ```
      * The components of the normalized tensor add up to 1.
      * The context vector for token 1 is then computed by multiplying the asociated attention weights with the
        corresponding token embeddings and summing them up.  ``` context_vector_token_1 = attention_weight_token_1[0] *
        token_e_0 + attention_weight_token_1[1] * token_e_0 + ...  attention_weight_token_1[n] * token_e_n ```
    * Transforming a `m` dimensional vector to `n` dimensions requires multliplication with a `m x n` matrix.
      * A vector, in PyTorch parlance, is represented as  list of numbers, i.e, vec = [x, y, z, ...]. It only has one
        dimension, i.e., the number of elements in the list.
      * Transforming a vector is achieved as,
        * x' = x * T, where T is a `m x n` sized matrix, or
        * x' = transpose(T) * x, where transpose(T) is a `n x m` sized matrix.
        * Essentially, the act of transformation can ve viewed as trnasforming "every" component of the incoming vector
          weighed differently to compute the components in the new outgoing vector.
