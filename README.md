#Data Sampler

Folder "datasets" contains the set data files.\
To generate data samples, execute file ```samples/file_sampler.py```.

##Sample generation definition
Array "datasets" defines how to generate samples for each data set. It is defiend as followed:
[1,2,3,4]
1. data set file name
2. if attributes are separated by whitespace instead of a comma
3. column names to delete
4. label, to use ase inilier class
5. QueryPoolMode

The first row of each data file should contain attribute names, where "class" denotes the column containing the class labels. 


