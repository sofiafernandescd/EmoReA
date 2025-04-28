# EmoReA Frontend

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)

This repository contains the React-based user interface for the EmoReA (Emotion Recognition Assistant) system. It provides a user-friendly way to interact with the backend API for emotion analysis.

## Features

-   Upload various file types (text, audio, image, video).
-   Submit files for emotion analysis to the EmoReA backend.
-   Display the emotion analysis results in a clear format.
-   Provides a chat interface to discuss the analysis with an AI assistant.

## Installation

1.  Ensure you have Node.js and npm (or yarn) installed on your system.
2.  Clone the main EmoReA repository:
    ```bash
    git clone https://github.com/sofiafernandescd/EmoReA
    cd emorea-frontend
    ```
3.  Install the dependencies:
    ```bash
    npm install
    # or
    yarn install
    ```

## Running the Frontend

For local development:

```bash
npm start
# or
yarn install
```

This will start the development server, usually accessible at http://localhost:3000. The frontend is configured to communicate with the backend, which by default is expected to be running at http://localhost:8000. You might need to adjust the backend API URL in your frontend configuration (e.g., in environment variables or a configuration file) if your backend is running elsewhere.

## Building for Production
```bash
npm run build
# or
yarn build
```
This command creates an optimized production build of the application in the build directory. You can then deploy the contents of this directory to a static hosting service like Firebase Hosting, Netlify, Vercel, or serve it using a web server like Nginx.

## Contributing
[Link to your contributing guidelines, if any]

## License
This project is licensed under the MIT License - see the LICENSE file for details.