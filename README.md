---
type: "project" # DON'T TOUCH THIS ! :)
date: "2020-05-16" # Date you first upload your project.
# Title of your project (we like creative title)
title: "This is an example project page which serves as a template"

# List the names of the collaborators within the [ ]. If alone, simple put your name within []
names: [Pilar Lopez Maggi, Gonzalo Giordano, Ana Pavlova Contreras, Santiago D'hers]

# Your project GitHub repository URL
github_repo: [github.com/sdhers/Mice-Behavioral-Analysis]

# List +- 4 keywords that best describe your project within []. Note that the project summary also involves a number of key words. Those are listed on top of the [github repository](https://github.com/PSY6983-2021/project_template), click `manage topics`.
# Please only lowercase letters
tags: [mice, tracking, machine learning, python]

# Summarize your project in < ~75 words. This description will appear at the top of your page and on the list page with other projects.

Our project aims to develop different methods for the analysis of behavior in mice (in this case, exploration of an object) to determine which is the best approach to this kind of study.
The accurate measurement of these behaviours is crucial for the study of neurodegenerative pathologies, such as Alzheimer’s disease.
We were able to implement and compare three increasingly complex methods to determine exploration time.
 * Manual labeling.
 * Motion tracking and data analysis using a custom algorithm.
 * Training a Machine Learning algorithm on our labeled data.

# If you want to add a cover image (listpage and image in the right), add it to your directory and indicate the name
# below with the extension.
image: "DLC_example.JPG"
---
<!-- This is an html comment and this won't appear in the rendered page. You are now editing the "content" area, the core of your description. Everything that you can do in markdown is allowed below. We added a couple of comments to guide your through documenting your progress. -->

## Project definition

### Tools

Our project will rely on the following technologies:

 * Python, to write the scripts used to label and analyze our data.
 * Jupyter Notebooks, to present our results in a clear and readable format.
 * DeepLabCut, to track movements of mice in our videos.

### Data

We worked on videos obtained with C57 mice during a Novel Object Recognition experiment developed at IFIBYNE - UBA, CONICET. The videos were first processed by DeepLabCut pose estimation software.

### Deliverables

At the end of this project, we will have:
 - A script to simplify the manual labeling of videos (including features to quickly label succesive frames by holding down a key and to go back if the user has made a mistake).
 - A Jupyter Notebook where the labeled data and the tracked positions are imported and processed, and where each of our exploration detection methods is applied and compared to the others.
 - Documentation on how to use our labeling script, Deep Lab Cut and each of our exploration detection methods.

## Results

### Progress overview

During the first week, we learned the basic tools which then allowed us to work on our project and collaborate easily: basic Bash commands, Git/GitHub and Python tools for data analysis. In the course of the following weeks, we defined the scope of our project and implemented each of our ideas with the help of our TAs.

### Tools we learned during this project

 * **Proper usage of version control systems for collaboration**: We learned how to properly use Git and Github to simplify collaboration between different team members.
 * We learned how to implement python scripts to read and process mice positions and label frames using spyder.
 * Finally, we could use the positions and labels to train a Random Forest (Machine Learning) model to predict labels on new videos.

### Results

#### Video labeling script

We developed different scrpts to be able to process the video information and label the frames with ease, you can find them in Video_processing/Label_Videos.py

#### Motion tracking using DLC

Once we got the positions from DLC, we could filter them according to distance and orientation towards the objects.

#.
image: "Criteria.png"
---

#### Applying and comparing each method

The most important part of our project is contained in exploration_detection.ipynb. To start with, we import the labels and the tracked data for each video, and we separate a video to use later to test the model. We then use the Random Forest model to process the positions and the given labels, and we test the unseen video using both the labels obtained manually, from the model and using the distance-orientation algorithm.

#.
image: "FrameperFrame.png"
---


#### Deliverable 3: Instructions

 To be made available soon.

## Conclusion and acknowledgement

