
# PYMC3 for economics and finance

In this repository, you will find all the materials I used during the workshop. You will also stumble upon some examples on topics I just barely covered in the slides.

Also bear in mind: you should be able to run everything within your IDE through your terminal (not the Python shell itself). Self containment and reproducibility are practices that, at worst, can't harm you and at best will be mandatory for any data scientist in the future. 

If you were to need it, you can still run the scripts, provided you know the adequate inputs.


## Scripts folder and output

If you followed the library installation instructions (`requirements.txt` and `arguments.txt`) steps correctly, the directory path to save your output in should now be easily created: 

```bash
cd . #Should be the main (root) directory of this repo
mkdir output

# Make sure to save the output path as a variable
# In some scripts, you will input this path as a variable
# It will be further useful if we need to create a set of separate folders later on

output_path= $PWD/ouput

```

### Script inputs

Now you just need to source your scripts in this manner:

```bash

python thescript.py <<< output_path [other_args]

``` 

If needed, you can always loop through the names and arguments in the `arguments.txt` to run them all. You should still try to read the python scripts and piece out the delimited sections.
 
 
 

