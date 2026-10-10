# Agent.md — Potato Doctor AI
## Existing Workspace Training + Vercel Deployment

### 1. Main Objective

Build a complete AI-based Potato Leaf Disease Detection website using the dataset already stored inside the currently open Antigravity project folder.

**Do not create a new project elsewhere. Do not download the dataset again. Do not ask for the dataset path unless the existing workspace inspection fails.**

The application must:

- Train a new Potato disease classification model using the existing dataset.
- Detect Potato Early Blight, Potato Late Blight, and Potato Healthy leaves.
- Provide evidence-based disease treatment and prevention guidance.
- Display Tomato, Corn, and Pepper as Coming Soon only.
- Be prepared for deployment on Vercel.
- Run and be tested locally before deployment.

Do not reuse the old broken `.pkl` models.

### 2. Inspect the Currently Open Workspace

Before changing any files:

1. Inspect the current project root.
2. Locate the dataset within the current workspace, including its subdirectories.
3. Identify the actual image folders and their class labels.
4. Count the images in each Potato class.
5. Confirm that all three required classes are present.
6. Inspect existing project code and configuration files.
7. Identify existing frontend frameworks, dependencies, and Git configuration.
8. Preserve the original dataset and unrelated project files.
9. Show a short report of the detected dataset structure and image counts.

Do not move or delete the dataset.

Do not duplicate a large dataset unnecessarily.

Do not assume folder names or invent missing classes.

If the dataset is already extracted, use it directly.

### 3. Current Crop Scope

Only Potato disease detection is available in this version.

Required classes:

- Potato Early Blight
- Potato Late Blight
- Potato Healthy

Verify the actual folder names and map them correctly.

The website must also show:

- Tomato — COMING SOON
- Corn — COMING SOON
- Pepper — COMING SOON

Do not train or integrate Tomato, Corn, or Pepper models yet.

Do not allow these upcoming crops to generate predictions.

### 4. Training Requirements

Use the dataset found in the current workspace.

Create training scripts inside the existing project, preferably under `training/`, without creating another project root.

Use Python and TensorFlow/Keras.

Prefer MobileNetV2 transfer learning as the initial approach. Use another appropriate image-classification architecture only when justified by the environment and evaluation results.

Training requirements:

1. Verify image labels and class counts.
2. Identify corrupted and unreadable images.
3. Check for duplicate images and data leakage where practical.
4. Split data into training, validation, and test sets, starting with 70%, 15%, and 15%.
5. Keep the test set untouched until final evaluation.
6. Use the correct model-specific preprocessing.
7. Apply realistic augmentation only to training images.
8. Use early stopping and model checkpointing.
9. Monitor validation loss and accuracy.
10. Save the best validated model.
11. Save the exact class mapping and preprocessing configuration.
12. Generate a training and evaluation report.

Never claim 100% accuracy or fabricate metrics.

If GPU access is unavailable, select reasonable batch sizes and image dimensions for the available hardware.

Do not retrain repeatedly without investigating poor results.

### 5. Model Artifacts

Save the newly trained model and configuration in the existing project, using a structure such as:

- `models/potato_model.keras`
- `models/class_mapping.json`
- `models/model_config.json`
- `reports/dataset_summary.json`
- `reports/classification_report.json`
- `reports/confusion_matrix.png`

Adapt paths to the current workspace where appropriate.

Do not commit the entire dataset to GitHub.

Do not place large model files in the frontend's public directory.

Before deployment, check model size, runtime dependencies, cold-start behavior, and the target platform's deployment limits.

### 6. Evaluation and Reliable Predictions

Evaluate the final model on unseen test images.

Report:

- Test accuracy
- Precision and recall for each class
- F1-score for each class
- Macro F1-score
- Confusion matrix
- Misclassified examples where practical

Ensure that Early Blight, Late Blight, and Healthy use the correct labels.

Use exactly the same image preprocessing during inference as during training.

Prediction requirements:

- Never hardcode Early Blight as the default result.
- Do not show a prediction when no image has been uploaded.
- Clear the previous result when a new image is uploaded.
- Reject invalid and corrupted images.
- Handle missing model files and inference errors explicitly.
- Do not invent confidence values.
- Do not treat high model confidence as proof of a correct diagnosis.
- Do not silently classify unrelated images as confirmed Potato diseases.

Evaluate performance on representative non-Potato images if implementing an image-relevance check.

Document the limitations of PlantVillage-based training when applied to real-world field photographs.

### 7. Vercel Deployment Architecture

**The final website must be deployable through Vercel.**

First inspect the existing workspace and select the simplest architecture that is genuinely compatible with Vercel.

Preferred approach:

- Frontend: React and Vite, or the existing compatible frontend framework.
- Frontend hosting: Vercel.
- Backend: use a Vercel-compatible Python function only if the trained model, required libraries, execution time, memory, request size, and model-loading behavior fit the applicable limits.
- If TensorFlow inference does not fit Vercel's limits, deploy the Python inference API to a compatible external Python hosting service and deploy the frontend to Vercel.

Do not force a large TensorFlow model into an incompatible Vercel serverless function.

Do not pretend that a deployment is complete when the backend cannot run.

Keep the deployment architecture as simple as possible and explain any external backend requirement.

### 8. Model Size and Hosting

Before choosing where inference runs:

1. Measure the trained model's size.
2. Check the selected platform's current function bundle, runtime, memory, execution-time, and request-size limits.
3. Verify the availability of TensorFlow and its dependencies in the selected runtime.
4. Estimate model-loading time and cold-start behavior.
5. Choose a compatible hosting method.

Do not upload the dataset to Vercel.

Do not place private model files or secrets in publicly served frontend assets.

If the model cannot fit within Vercel's limits, keep the frontend on Vercel and use an external inference API.

### 9. Treatment Advice with OpenAI API

After a Potato prediction, provide treatment and prevention guidance using the OpenAI API when configured.

Use verified agricultural information as the source of treatment recommendations.

The image classifier determines the predicted class. The language model explains the result and provides guidance; it must not invent or change the classifier's prediction.

Provide:

- Disease explanation
- Common symptoms
- Immediate practical steps
- Prevention advice
- Safe, evidence-based treatment guidance
- When to consult an agricultural expert
- Reliable information sources
- A clear diagnostic disclaimer

Do not invent pesticide products, dosages, mixing instructions, or waiting periods.

Use a curated treatment knowledge base from reputable agricultural extension services, universities, or plant pathology resources.

If the OpenAI API is unavailable or no API key is configured, display the curated treatment guidance instead.

Keep `OPENAI_API_KEY` on the server side only. Never expose it in frontend code, public environment variables, or Git commits.

If Vercel is hosting the API route, verify that the route and all its dependencies fit Vercel's limits. If not, host the API separately.

### 10. Website Requirements

Create a polished and responsive website named **Potato Doctor AI**.

Include:

- Homepage and navigation
- Crop selection dashboard
- Potato image upload and preview
- Prediction button and loading state
- Predicted disease and confidence information
- Treatment and prevention guidance
- Reset button
- Helpful error messages
- Mobile and desktop layouts
- About and responsible-use sections

Crop dashboard:

**Potato**
- Status: Available
- Fully working disease detection

**Tomato**
- Status: Coming Soon
- No prediction functionality

**Corn**
- Status: Coming Soon
- No prediction functionality

**Pepper**
- Status: Coming Soon
- No prediction functionality

Do not show fake predictions for upcoming crops.

### 11. API and Configuration

If a separate inference backend is needed, implement suitable endpoints such as:

- `GET /health`
- `GET /classes`
- `POST /predict`
- `POST /advice`

Validate uploaded files and requests.

Return structured JSON and appropriate error codes.

Use environment variables for the production API URL.

Never hardcode localhost URLs in production.

Configure CORS securely for the deployed frontend.

Create `.env.example` without actual secrets.

Do not expose stack traces, internal paths, or credentials.

### 12. Testing

Test the application locally before deployment.

Verify:

- Valid Early Blight image
- Valid Late Blight image
- Valid Healthy image
- Missing image
- Invalid file type
- Corrupted image
- Non-Potato image handling
- Missing or incompatible model
- OpenAI API unavailable
- New image replacing an old prediction
- Tomato, Corn, and Pepper Coming Soon cards
- Frontend production build
- Backend health and prediction endpoints
- Production API connectivity after deployment

Use real test images and report actual results.

Do not hardcode predictions to make tests pass.

### 13. Vercel Configuration and Deployment

Inspect the existing project before creating or modifying deployment files.

Configure the Vercel project for the actual frontend framework and repository structure.

Requirements:

- Correct build command
- Correct output directory
- Correct root directory
- Correct environment variables
- Correct production API URL
- Secure handling of secrets
- Compatible routing and error handling

Do not add unnecessary configuration files.

Do not overwrite working configuration without checking its purpose.

Run the production build locally.

Deploy only after verifying the frontend and the required backend architecture.

If deployment requires credentials or account access that are unavailable, prepare the project and provide the exact remaining steps without claiming success.

### 14. Final Report

At completion, report:

1. The dataset directory used from the currently open workspace.
2. Actual image counts for the three Potato classes.
3. Training configuration and model architecture.
4. Actual test metrics.
5. Files created or modified.
6. Local test results.
7. How to run the application.
8. Vercel configuration and deployment status.
9. Whether a separate Python backend is required.
10. Any remaining limitations.

Do not claim that training, testing, or deployment succeeded unless it was actually completed and verified.

### Final Priority

Use the dataset already in the open workspace. Train and evaluate the Potato model properly. Build a reliable application, keep Tomato, Corn, and Pepper as Coming Soon, protect API keys, and deploy the frontend to Vercel using an architecture that genuinely supports the model.