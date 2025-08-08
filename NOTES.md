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


## Deep Lerning

* Sigmoid is represented as $`f(x)=1/(1+\exp(-x))`$, which is the same as
  $`f(x)=\exp(x)/(1+\exp(x))`$. As x approaches infinity, `f(x)` approaches 1.
  `1` can be replaced by any other positive value, say `L`, which then becomes
  the upper limit. The range of the sigmoid function is `[0, 1]` or `[0, L]`, and
  the midpoint is `0.5`.