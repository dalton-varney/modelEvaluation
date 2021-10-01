# Model Evaluation App
This code is based on the video streamer from https://github.com/alwaysai/video-streamer

## Requirements
* [alwaysAI account](https://alwaysai.co/auth?register=true)
* [alwaysAI Development Tools](https://alwaysai.co/docs/get_started/development_computer_setup.html)

# Evaluate Your model
1. Change ```obj_detect``` in app.py to point to your model (username/model)
2. Put your zipped dataset into the application folder (Must be PASCALVOC)
3. Change ```zipped_data``` constant in app.py to point to include your dataset file name
4. Run the steps below to configure, install, and start
5. You can change the speed parameters as well with: ```slideShowSpeed```

## Usage
Once the alwaysAI tools are installed on your development machine (or edge device if developing directly on it) you can install and run the app with the following CLI commands:

To perform initial configuration of the app:
```
aai app configure
```

To prepare the runtime environment and install app dependencies:
```
aai app install
```

To start the app:
```
aai app start
```

To change the computer vision model, the engine and accelerator, and add additional dependencies read [this guide](https://alwaysai.co/docs/application_development/configuration_and_packaging.html).


## Support
* [Documentation](https://alwaysai.co/docs/)
* [Community Discord](https://discord.gg/z3t9pea)
* Email: support@alwaysai.co
