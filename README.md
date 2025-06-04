# Introduction Pages
The rollout of surveillance cameras in cities, transport terminals, and public places over the last decade has led to huge volumes of video data. Such surveillance systems are vital to public safety and security when they are used to observe suspicious activities or intruders, speeding vehicle and possibly alerts for emergency responses. The volume of footage generated daily does not allow for "eyes-on" monitoring to be practical (or responsive) and may result in missed events and late response times.

With such difficulties, approaches for the analysis of intelligent videos have been receiving more and more attention. Real-time object detection models like YOLOv8 have shown impressive results in being able to accurately and quickly detect key objects, especially people and vehicles, which are at the core of many surveillances use cases. However, classical object detection only has bounding boxes with class and does not have the descriptive context which is very important for incident analysis, reporting and automatic alerts generation.

We propose a comprehensive, end-to-end intelligent surveillance framework that integrates YOLOv8 for detecting people and vehicles, alongside the BLIP model to produce concise and human-readable scene descriptions. For each frame or detected event, the system automatically generates structured logs which record both the detection results and the corresponding natural language captions. These logs can easily be used for further analysis, incident searching, and get alerts in real time, helping to automate things, so to free more time from analysts to do analysis on the incoming security events.

We validate the effectiveness of our approach on datasets including the AI City Challenge, with detection accuracy in term of mean Average Precision (MAP) and the quality of generated captions in terms of BLEU and ROUGE metrics. The integration of advanced object detection and image captioning techniques enhances the value of surveillance data, making automatic monitoring more intuitive and insightful for practical applications.




