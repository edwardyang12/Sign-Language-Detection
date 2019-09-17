# Sign-Language-Detection

TODO:
make a cool webapp
design an algorithm (look into approaches such as detecting individual fingers)



Finished: got a basic classifier to work

References: https://towardsdatascience.com/all-the-steps-to-build-your-first-image-classifier-with-code-cf244b015799
http://szeliski.org/Book/drafts/SzeliskiBook_20100903_draft.pdf


Old classifier:
References https://medium.com/datadriveninvestor/building-an-image-classifier-using-tensorflow-3ac9ccc92e7c

Train on windows: python3 -m scripts.retrain --bottleneck_dir=tf_files/bottlenecks --how_many_training_steps=500 --model_dir=tf_files/models/ --summaries_dir=tf_files/training_summaries/"%ARCHITECTURE%" --output_graph=tf_files/retrained_graph.pb --output_labels=tf_files/retrained_labels.txt --ARCHITECTURE="%ARCHITECTURE%" --image_dir=tf_files/fingers

Run classify on windows: python3 -m scripts.label_image --image=tf_files/test_fingers/middle_finger3_test.jpg

*run inside tensorflow for poet directory
