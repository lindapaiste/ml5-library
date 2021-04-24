import {createModel} from "./modelBuilder";
import {ARGS} from "./argumentValidator";
import * as handposeCore from "@tensorflow-models/handpose";
import * as faceapiCore from "face-api.js";

const handpose = createModel({
    acceptsArgs: {
        video: ARGS.video,
        options: ARGS.options,
        callback: ARGS.callback,
    },
    defaults: {
        //handpose doesn't actually define defaults!!
    },
    buildModel: (args) => {
        return handposeCore.load(args.options);
        
        // TODO: handle video
    },
    methods: {
        
    }
})

const faceapi = createModel({
    acceptsArgs: {
        video: ARGS.video,
        options: ARGS.options,
        callback: ARGS.callback,
    },
    defaults: {
        withLandmarks: true,
        withDescriptors: true,
        minConfidence: 0.5,
        withTinyNet: true,
        withFaceExpressions: false,
        withAgeAndGender: false,
        MODEL_URLS: {
            Mobilenetv1Model:
                "https://raw.githubusercontent.com/ml5js/ml5-data-and-models/main/models/faceapi/ssd_mobilenetv1_model-weights_manifest.json",
            TinyFaceDetectorModel:
                "https://raw.githubusercontent.com/ml5js/ml5-data-and-models/main/models/faceapi/tiny_face_detector_model-weights_manifest.json",
            FaceLandmarkModel:
                "https://raw.githubusercontent.com/ml5js/ml5-data-and-models/main/models/faceapi/face_landmark_68_model-weights_manifest.json",
            FaceLandmark68TinyNet:
                "https://raw.githubusercontent.com/ml5js/ml5-data-and-models/main/models/faceapi/face_landmark_68_tiny_model-weights_manifest.json",
            FaceRecognitionModel:
                "https://raw.githubusercontent.com/ml5js/ml5-data-and-models/main/models/faceapi/face_recognition_model-weights_manifest.json",
        }
    },
    buildModel: async ({options}) => {
        // initialize operation sets a property detectorOptions that is separate from model
        Object.keys(options.MODEL_URLS).forEach(item => {
            if (MODEL_NAMES.includes(item)) {
                options.MODEL_URLS[item] = this.getModelPath(options.MODEL_URLS[item]);
            }
        });

        const {
            Mobilenetv1Model,
            TinyFaceDetectorModel,
            FaceLandmarkModel,
            FaceRecognitionModel,
            FaceLandmark68TinyNet,
        } = options.MODEL_URLS;

        const model = faceapiCore;


        if (options.withTinyNet === true) {
            this.detectorOptions = new faceapiCore.TinyFaceDetectorOptions({
                // TODO: property name minConfidence is not supported -- should this be scoreThreshold?
                minConfidence: options.minConfidence,
                inputSize: 512,
            });
        } else {
            this.detectorOptions = new faceapiCore.SsdMobilenetv1Options({
                minConfidence: options.minConfidence,
            });
        }

        // check which model to load - tiny or normal
        if (options.withTinyNet === true) {
            await model.loadFaceLandmarkTinyModel(FaceLandmark68TinyNet);
            await model.loadTinyFaceDetectorModel(TinyFaceDetectorModel);
        } else {
            await model.loadFaceLandmarkModel(FaceLandmarkModel);
            await model.loadSsdMobilenetv1Model(Mobilenetv1Model);
        }
        await model.loadFaceRecognitionModel(FaceRecognitionModel);

        return model;
    }
    
})