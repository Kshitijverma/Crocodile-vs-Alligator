import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import ReptileClassifier from './components/ReptileClassifier';
import * as serviceWorker from './serviceWorker';

ReactDOM.render(<ReptileClassifier />, document.getElementById('root'));

serviceWorker.unregister();
