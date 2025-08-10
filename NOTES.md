# Notes

## Python:

* Raw strings i.e., `r""` are treated as-is. In other words, escape sequences
like `\n` or `\t` will not be expanded. The only exception is a backslash, `\`,
which still needs to be backslashed to be interprted correctly i.e., `\\`.
* When using regular expression for splitting, the `()` can be used to capture
the separator itself. 
* In `re.split(r'([\.,:;\?_!]|--|\s)', "Hello, World. I am alive?")`, all
separators are enclosed within `()`. Also, multiple separators are separated
using `|`. Punctuations are grouped inside `[]` that signifies on one of.
Backslash befor `?` and `.` are possibly not required, but it doesn't hurts.
* In `re.sub(r'\s+([,.?!"()\'])', r'\1', text)`, we are essentially replacing
the occurence ` ;` with `;`. The `\1` is a reference to the first capture group.
The replacement is necessary because in its absence we will get teh text
`Hello World !` as opposed to `Hello World!`.


## Deep Learning

* Sigmoid is represented as $`f(x)=1/(1+\exp(-x))`$, which is the same as
  $`f(x)=\exp(x)/(1+\exp(x))`$. As x approaches infinity, `f(x)` approaches 1.
  `1` can be replaced by any other positive value, say `L`, which then becomes
  the upper limit. The range of the sigmoid function is `[0, 1]` or `[0, L]`,
  and the midpoint is `0.5`.
* The function `binary_cross_entropy` returns a scalar.  It is used to compare    
  `input` probabilities (in the range [0, 1]) to `target` class labels (either
   0 or 1).
* A linear regression is represented by $`y = w*x + b`$, where `w` is weight and
  `b` is bias. The snippet below shows some examples,
  ```
  w*x -> [128, 784] * [784, 1] -> [128, 1]
  w*x -> [64, 128] * [128, 1] -> [64, 1]
  ```
  * The thing to keep in mind is that we are multiplying `w` with the incoming
  vector `x`, not the other way around.
  * The first dimension of the weight vector represents the number of nodes in
  the next layer of the neural network.
  * The second dimension of the weight vector represents the dimension of the
  input vector.
