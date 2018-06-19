# kitchentabletools-python

A collection of my Python scripts to GSD. Inspired by the collection of [kitchentabletools](https://github.com/petersheridandodds/kitchentabletools) from [Peter Dodds](http://www.uvm.edu/pdodds/).

Pull requests gladly accepted :)

## Design

This is not a proper python package, but rather a collectin of scripts to be used/edited/etc.
A pile of scripts is lower cost to edit and push than a package!

## Use

Clone this repo in your local tools directory.
That will look like:

```sh
cd ~/tools/python
git clone git@github.com:andyreagan/kitchentabletools-python.git kitchentable
```

And then in Python:

```python
import os
import sys
# some people say modifying path is bad practice
# forget those people, this package's main purpose is GSD
sys.path.append(os.path.join(os.environ.get("HOME"),"tools","python"))
from kitchentable.dogtoys import mysavefig
```


