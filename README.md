# pythonBenchmark

Python models used for joel's benchmark tests to compare with R results of Veronica/Sonja

Before running the serving scripts, you will need to extract all data in "data" folder, and run the forest_train, once with n_estimators=50 and once with n_estimators=500.

## Run docker images from docker hub:
If you just want to run the previously created images, they are hosted on docker hub at matleo/pythonbenchmark. So you can just execute:
* docker run -p 8003:5000 matleo/pythonbenchmark:baseline
* docker run -p 8004:5000 matleo/pythonbenchmark:forest_50
* docker run -p 8005:5000 matleo/pythonbenchmark:forest_500
