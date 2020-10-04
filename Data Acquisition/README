### TASK

Download corpora:
	- Argument Annotated Essays (version 2) : https://www.informatik.tu-darmstadt.de/ukp/research_6/data/argumentation_mining_1/argument_annotated_essays_version_2/index.en.jsp
	- Insufficiently Supported Arguments in Argumentative Essays: https://www.informatik.tu-darmstadt.de/ukp/research_6/data/argumentation_mining_1/insufficiently_supported_arguments/index.en.jsp
	- Opposing Arguments in Persuasive Essays: https://www.informatik.tu-darmstadt.de/ukp/research_6/data/argumentation_mining_1/opposing_arguments_in_persuasive_essays/index.en.jsp


Corpora Unification: Each object represents an essay that contains:
	- Essay Identifier
	- Essay text
	- Confirmation bias as a boolean label
	- List of Paragraphs; each as a json object containing text, and the sufficient label
	- Premises, Claims and Major Claims each as a list of json objects, where each object contains text and span as tuple.

Preliminary Statistics: On the training split perform the following:
	- Number of essays, paragraphs, sentences, and tokens (use spaCy api for
	tokenization).
	- Number of major claims, claims, premises.
	- Number of essays with and without confirmation bias.
	- Number of sufficient and insufficient paragraphs (arguments).
	- Average number of tokens in major claims, claims, and premises.
	- The 10 most specific words in major claims, claims, and premises.


### Assignment Protocol:

What you get from us?
	- Material: Under materials folder you can find papers in which the required datasets were proposed.
	- A sample file: sample_output.json file represents the structure of the desired unification of the datasets.

What we expect from you?
	- Corpora: Under the data folder, including the three downloaded corpora and the unified data file.
	- Code: Under the code folder, including the python file to generate the unified data file and a Jupyter notebook containing the statistics.
	- Documentation: A PDF file that contains instructions how to reproduce the unified data file and explanation of the method you used to compute the most specific words of each of the argument units.

Assignment Grading:
	- (F) If the unification code does not work or produces incorrect output.
	- (B) If at least 50% of the statistics are correct.
	- (A) All statistics are correct, convincing way of computing the specific words of argument units, and clear documentation.

