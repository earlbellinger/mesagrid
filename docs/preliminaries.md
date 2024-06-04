# Preliminaries

The `mesagrid` package uses outputs from [MESA](https://docs.mesastar.org/) and [GYRE](https://gyre.readthedocs.io). Let's install them! An installation script can be found [here](https://gist.github.com/earlbellinger/5c79b175bcb6b4ea9e934d039c701990).


Once the installation script is finished, you should see: 


```
************************************************
************************************************
************************************************

MESA installation was successful

************************************************
************************************************
************************************************
```

Hooray! You are ready to get started. 

## Your first MESA run 

Let's copy a fresh work directory over: 

```language:bash
cp -R $MESA_DIR/star/work .
cp -R $MESA_DIR/star/defaults/*.list work
cd work
./mk
./rn
```

That should have worked. If so, we're ready to compute some models for asteroseismology. 

For a slower pace of the installation and the following tutorial, see the recorded tutorial [here](https://github.com/earlbellinger/mesa-gyre-tutorial-2022).

Let's change the mass, initial chemical composition, and mixing length parameter so that we roughly get a solar-like model. You can copy over the values from my inlist given in the `src/` directory. I have also instructed MESA to output [FGONG](https://phys.au.dk/~jcd/ASTEC_results/file-format.pdf) whenever a profile file is generated. 

Let's also uncomment `delta_nu` and `nu_max` from `history_columns.list`, and `brunt_N` and `lamb_S` from `profile_columns.list`. 

Now when we run this we should get some output in the LOGS directory. Let's analyze that with python now. 

## Compute frequencies with GYRE

Now let's calculate some frequencies. An example GYRE inlist awaits you in the `src/` directory. Copy that over to your `LOGS/` directory and run it with GYRE: 

```
    cp ../src/gyre.in LOGS
    cd LOGS
    $GYRE_DIR/bin/gyre gyre.in
```

I've prepared a script in the `src/` directory that will allow us to efficiently obtain the frequencies of all our stellar models. Let's run that in a loop: 

```
    for FGONG in *.FGONG; do
        ../../src/gyre6freqs.sh -i $FGONG -f
    done
```

or even better, using `xargs` in parallel: 

```
find . -name "*.FGONG" -print0 | xargs -0 -P 8 -I{} ../../src/gyre6freqs.sh -i {} -f -t 1
```


## A small grid 

Let's now try to run a grid of tracks with different masses. We'll begin by copying our work over to a new directory and removing the logs files: 

```
cp -R work grid
cd grid
rm -rf LOGS
```

Now let's edit `inlist_project` to stop the run at a central hydrogen fraction of 0.1, and also specify the `log_directory = '1'` in the `&controls` block. 

Afterwards, we are ready to use `shmesa` to run a small grid of models: 

```
mkdir grid
for M in 0.7 0.8 0.9 1.0 1.1 1.2; do
    echo "Running mass $M"
    shmesa change inlist_project initial_mass $M
    shmesa change inlist_project log_directory "grid/M_'$M'"
    ./star inlist_project

    cd $M
    for GYRE in *.GYRE; do
        ../../src/gyre6freqs.sh -i $FGONG -f -t $OMP_NUM_THREADS
    done
    cd -
done 
```


Additional useful resources
---

- [tomso](https://tomso.readthedocs.io/): Tools for Models of Stars and their Oscillations
- [astero](https://cococubed.com/mesa_market/astero_module.html): Overview of the `astero` module 
- [tulips](https://astro-tulips.readthedocs.io/): Tools for Understanding the Lives, Interiors, and Physics of Stars
