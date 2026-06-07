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
We appreciate all contributions! Feel free to open an issue or submit a pull request to suggest improvements. All contributors will be formally acknowledged and highlighted in this section.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project, code, or findings in your research, please cite the corresponding publication(s) below:

### 1. Master's Thesis
> Condesso, S. F. (2025). *Emotion recognition in multimedia content* (Master's thesis, Instituto Superior de Engenharia de Lisboa). Repositório Institucional do IPL. http://handle.net

```bibtex
@mastersthesis{thesis2025emotion,
  author       = {Condesso, Sofia Fernandes and Ferreira, Artur Jorge and Leite, Nuno Miguel da Costa de Sousa},
  title        = {Emotion recognition in multimedia content},
  school       = {Instituto Superior de Engenharia de Lisboa},
  year         = {2025},
  type         = {Master's thesis},
  url          = {http://handle.net}
}
```

### 2. Conference Papers

#### Facial Emotion Recognition (2026)
> Condesso, S., Ferreira, A. J., & Leite, N. (2026). Facial Emotion Recognition: A Comparative Study with Cross-Corpus and Multi-Corpus Training. In *Proceedings of the Conference*.

```bibtex
@inproceedings{inproceedings,
author = {Condesso, Sofia and Ferreira, Artur and Leite, Nuno},
year = {2026},
month = {03},
pages = {},
title = {Facial Emotion Recognition: A Comparative Study with Cross-Corpus and Multi-Corpus Training},
doi = {10.5220/0014639300004067}
}
```

#### User Emotion Recognition (2025)
> Condesso, S., Ferreira, A. J., & Leite, N. (2025). User Emotion Recognition from Speech and Text Messages with Machine Learning Techniques. In *Proceedings of the Conference*.

```bibtex
@inproceedings{inproceedings,
author = {Condesso, Sofia and Ferreira, Artur and Leite, Nuno},
year = {2025},
month = {10},
pages = {},
title = {User Emotion Recognition from Speech and Text Messages with Machine Learning Techniques}
}
```