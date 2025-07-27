# Notes

* Raw strings i.e., `r""` are treated as-is. In other words, escape sequences
like `\n` or `\t` will not be expanded. The only exception is a backslash, `\`,
which still needs to be backslashed to be interprted correctly i.e., `\\`.
* When using regular expression for splitting, the `()` can be used to capture
the separator itself. 
** In `re.split(r'([\.,:;\?_!]|--|\s)', "Hello, World. I am alive?")`, all
separators are enclosed within `()`. Also, multiple separators are specified
using `|`. Additionally, punctuations are grouped inside `[]`.
