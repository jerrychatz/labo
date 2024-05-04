# 🗣️ Language in a Bottle 🍾

This repository serves as a first step to explore Language Guided Concept Bottleneck Models. Great appreciation is extended to the original creators for their exceptional work. You can find the original repository hosted under the following GitHub account:

- [Original Repository](https://github.com/YueYANG1996) - Located within the `LaBo` folder.

For a comprehensive understanding of the concepts and methodologies implemented in this project, refer to the paper available at:

- [Language Guided Concept Bottleneck Models](https://arxiv.org/abs/2211.11158)

This paper provides the foundational theory and detailed explanations of the techniques and models utilized in this repository.



## Datset
To download the CIFAR-10 dataset, follow these steps:

1. **Download the CIFAR-10 Python Version**:
   - Visit the CIFAR-10 dataset page on the University of Toronto website: [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
   - Download the Python version of the CIFAR-10 dataset.

2. **Setup the Dataset Directory**:
   - After downloading, extract the contents and move the `cifar-10-batches-py` folder to your project directory under `LaBo/datasets/CIFAR10`.

3. **Configure the Python Script**:
   - Ensure that the `Config` class within your script correctly points to the new dataset location.
   - Example configuration:
     ```python
     class Config:
         dataset_path = "/path/to/your/LaBo/datasets/CIFAR10/cifar-10-batches-py/"
     ```

4. **Run the Processing Script**:
   - Execute the script to process the CIFAR-10 data by running:
     ```
     python3 process_cifar.py
     ```

This setup prepares the CIFAR-10 dataset for further experiments or processing as required by your project.

## Environment Setup
Follow these steps to set up and troubleshoot the required software for the `LaBo` project:

1. **Follow Project Instructions**:
   - Begin by following the initial setup instructions as detailed in the `LaBo` folder.

2. **Install Required Packages**:
   - Install necessary Python packages using pip:
     ```
     pip3 install openai python-dotenv
     ```

3. **Address Installation Issues**:
   - If you encounter any issues during the installation, refer to the discussion and solutions provided by the project community:
     - General installation issues have been addressed here: [Installation Issues Resolved](https://github.com/YueYANG1996/LaBo/issues/14#issuecomment-1847983397).

4. **Modify Apricot Package**:
   - Should you encounter errors with the `apricot` package, follow the modification suggested by the project authors:
     - Specific instructions to resolve `apricot` errors are detailed here: [Apricot Package Fix](https://github.com/YueYANG1996/LaBo/issues/1#issuecomment-1583107414). As suggested, you might need to comment out certain lines in the `apricot` package files.

These steps should help you successfully set up and troubleshoot the software required for the `LaBo` project.

## Concept Generation
In order to create the initial concepts using GPT-3.5 Turbo, you first need to obtain an API key from OpenAI.

1. **Obtain an API Key**:
   - You need to create an account on OpenAI's platform and follow the process to obtain an API key.

2. **Setup Environment Variables**:
   - Create a `.env` file in the root directory of your project.
   - Add the following line to the `.env` file:
     ```
     OPENAI_API_KEY=INSERT_YOUR_API_KEY
     ```

3. **Configure the Helper Script**:
   - Open the `helper.yaml` file.
   - Change the OpenAI model and modify the prompts as required.

4. **Run the Concept Generation Script**:
   - Execute the script by running the following command in your terminal:
     ```
     python3 concept_generation.py
     ```
   - This will generate a `concepts.json` file that contains sentences for each class label in the CIFAR-10 dataset.


## Concept Extraction
To perform concept extraction using the T5 fine-tuned model provided by the authors, follow these steps:

1. **Download the T5 Fine-tuned Model**:
   - Access the model through this Google Drive link: [Download T5 Fine-tuned Model](https://drive.google.com/file/d/1c1ax5J6gaItxHyIaTvuiduhSNB0LoVbn/view)
   - This link was shared in an issue on the repository, which can be found here: [GitHub Issue Link](https://github.com/YueYANG1996/LaBo/issues/22).

2. **Unzip the Downloaded File**:
   - After downloading, unzip the file to your desired directory.

3. **Run the Concept Extraction Script**:
   - Navigate to the directory containing the `concept_extractor.py` script.
   - Execute the script by running the following command in your terminal:
     ```
     python3 concept_extractor.py
     ```
   - This will generate a `concepts_t5_extracted.json` file that contains the specific, descriptive concepts extracted from the `concepts.json` file.

After creating the concept descriptions, you need to organize and compare them as follows:

1. **Rename the Concept File**:
   - Rename the generated concept file to `class2concepts.json`.

2. **Move the Renamed File**:
   - Move the renamed file to the appropriate directory in your project structure:
     ```
     mv class2concepts.json /path/to/your/LaBo/datasets/CIFAR10/concepts/class2concepts.json
     ```

3. **Comparison Setup**:
   - Ensure that all implementations remain constant to allow for an accurate comparison between the newly generated concepts and the original ones.

By following these steps, you can effectively compare the effectiveness and relevance of the generated concept descriptions against the baseline provided by the original concepts.

# Training LaBo
Training with the `LaBo` project involves a series of steps that are well-documented in the `README.md` file located in the `LaBo` directory. Here's how to proceed with setting up and customizing your training process:

1. **Follow General Training Instructions**:
   - Refer to the `README.md` file in the `LaBo` directory for detailed instructions on how to set up and execute the training process.

2. **Adjust Training Configuration**:
   - To modify training parameters such as `max_epochs`, edit the configuration file:
     ```
     LaBo/cfg/asso_opt/CIFAR10/CIFAR10_base.py
     ```
   - Adjust the `max_epochs` parameter within this file to suit your training needs.

3. **Customize Training Script**:
   - Modify the `labo_train.sh` script to match your specific training setup:
     ```bash
     python main.py --cfg cfg/asso_opt/$2/$2_$1shot_fac.py --work-dir exp/asso_opt/$2/$2_$1shot_fac --func asso_opt_main 
     ```
   - Replace `$1` and `$2` with appropriate values that reflect the number of shots and the dataset or configuration specifics.

4. **Customize Testing Script**:
   - For testing, update the `labo_test.sh` script as follows:
     ```bash
     python main.py --cfg $1 --func asso_opt_main --test --cfg-options bs=512 ckpt_path=$2
     ```
   - Here, `$1` should be replaced with the path to your configuration file, and `$2` with the path to the checkpoint file you wish to use for testing.

By following these steps and making the necessary adjustments, you can effectively train and test using the `LaBo` project, tailoring the process to meet your specific research or project needs.
