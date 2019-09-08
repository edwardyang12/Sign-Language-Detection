# Sign.Language.Detection-
New classifier:

TODO:
make a cool webapp
fix the algorithm/ make an algorithm 






Old classifier:
References https://medium.com/datadriveninvestor/building-an-image-classifier-using-tensorflow-3ac9ccc92e7c

Train on windows: python3 -m scripts.retrain --bottleneck_dir=tf_files/bottlenecks --how_many_training_steps=500 --model_dir=tf_files/models/ --summaries_dir=tf_files/training_summaries/"%ARCHITECTURE%" --output_graph=tf_files/retrained_graph.pb --output_labels=tf_files/retrained_labels.txt --ARCHITECTURE="%ARCHITECTURE%" --image_dir=tf_files/fingers

Run classify on windows: python3 -m scripts.label_image --image=tf_files/test_fingers/middle_finger3_test.jpg

*run inside tensorflow for poet directory
