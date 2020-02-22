# FADO
Framework for Aerostructural Design Optimization

## Motivation
Q: "What is my purpose?"

A: "You run other codes."

More seriously, the typical use case is:
you have a simulation code (CFD, FEA, etc.) that reads/writes its inputs/outputs from/to text file(s),
you want to wrap the execution of that code, for example to use it in an optimization,
but that code does not have a Python interface quite like what you would like...

FADO provides abstractions to make this job easier than writing specialized scripts for each application,
scripting is still required, but the resulting scripts should be easier to maintain/adapt/modify.

The design of the framework is centered around large scale applications (10k+ variables) and functions that are expensive to evaluate (compared to the cost of preparing input files).

## Abstractions
From the top down (and not a replacement for the docs):

- **Driver**: The class of objects eventually passed to an optimizer to wrap the execution steps required to evaluate functions and their gradients, a driver is therefore composed of "functions". Different applications benefit from drivers with different characteristics, for example the ExteriorPenaltyDriver can evaluate all its functions simultaneously.
- **Function**: An entity with one scalar output and any number of input "variables". Functions are further defined by the steps ("evaluations") required to obtain their value and possibly their gradient.
- **Variable**: The scalar or vector inputs of functions that are exposed to the optimizers.
- **Evaluation**: These wrap the calls to the external codes, they are configured with the input and data files, and the instructions, required to execute the code. "Parameters" can be associated with evaluations to introduce small changes to the input files (e.g. change a boundary condition in a multipoint optimization).
- **Parameter**: A numeric or text variable that is not exposed to the optimizer, they are useful to introduce small modifications to the input files to make a small number of template input files applicable to as many evaluations as possible.

## Interfacing with files
Function, Variable, and Parameter need ways to be written and read to or from files.
Any object implementing write(values,file) or read(file) can be used, four classes are provided that should cover most scenarios:

- **LabelReplacer**: Replaces any occurrence of a label (a string) with the value of a scalar variable or parameter.
- **PreStringHandler**: Reads(writes) a list of values separated by a configurable delimiter from(in) front of a label defining the start of a line (i.e. the line must start with the label).
- **TableReader/Writer**: Reads or writes to a section of a delimited table, rows outside of the table range do not need to be in table format, but those inside are expected to have the same number of columns, the examples should make it clear how to use these classes.

## Installation
Make the parent directory ("../") and FADO's ("./") reachable to Python, usually via PYTHON_PATH, `from FADO import *` should then work (provided the name of the directory was not changed).

## Usage
Have a look at the examples, example1 is a contrived example using the Rosenbrock function, the others are realistic uses of [SU2](https://su2code.github.io/).
Calls to external codes are made with `subprocess.call(..., shell=True)`, don't run optimizations as root :)

## License
[LGPL-3.0](https://www.gnu.org/licenses/lgpl-3.0.html)

