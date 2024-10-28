// Jenkinsfile
pipeline {
    agent any
    environment {
        DATABASE_URL = credentials('database_url')
    }
    stages {
        stage('Build') {
            steps {
                sh 'docker build -t forecast_app .'
            }
        }
        stage('Test') {
            steps {
                sh 'pytest tests/'
            }
        }
        stage('Deploy') {
            steps {
                sh 'docker-compose up -d'
            }
        }
    }
}