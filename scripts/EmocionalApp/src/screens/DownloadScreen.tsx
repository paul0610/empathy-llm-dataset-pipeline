/**
 * Pantalla de Descarga del Modelo IA
 * Basada en el prototipo UX de Stitch
 */

import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import RNFS from 'react-native-fs';
import { StackScreenProps } from '@react-navigation/stack';
import { Colors, Typography, Spacing } from '../constants/DesignSystem';
import { RootStackParamList, DownloadState } from '../types';
import CircularProgress from '../components/CircularProgress';
import Button from '../components/Button';
import { saveModelPath, getModelPath } from '../utils/storage';

type Props = StackScreenProps<RootStackParamList, 'Download'>;

const MODEL_URL =
  'https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf?download=true';

const DownloadScreen: React.FC<Props> = ({ navigation }) => {
  const [downloadState, setDownloadState] = useState<DownloadState>({
    progress: 0,
    speed: 0,
    bytesWritten: 0,
    totalBytes: 0,
    status: 'idle',
  });

  useEffect(() => {
    checkAndDownloadModel();
  }, []);

  const checkAndDownloadModel = async () => {
    try {
      // Verificar si el modelo ya existe
      const existingPath = await getModelPath();
      if (existingPath && (await RNFS.exists(existingPath))) {
        console.log('Model already exists, navigating to chat');
        navigation.replace('Chat', { modelPath: existingPath });
        return;
      }

      // Iniciar descarga
      await downloadModel();
    } catch (error) {
      console.error('Error checking/downloading model:', error);
      setDownloadState(prev => ({
        ...prev,
        status: 'error',
        error: 'Error al descargar el modelo. Por favor, intenta de nuevo.',
      }));
    }
  };

  const downloadModel = async () => {
    const destDir = `${RNFS.DocumentDirectoryPath}/models`;
    const destPath = `${destDir}/llama3-1b-q8.gguf`;

    try {
      // Crear directorio si no existe
      await RNFS.mkdir(destDir);

      setDownloadState(prev => ({ ...prev, status: 'downloading' }));

      let lastUpdate = Date.now();
      let lastBytes = 0;

      const job = RNFS.downloadFile({
        fromUrl: MODEL_URL,
        toFile: destPath,
        progress: (res) => {
          const now = Date.now();
          const timeDiff = (now - lastUpdate) / 1000; // segundos
          
          if (timeDiff >= 0.5) { // Actualizar cada 0.5 segundos
            const bytesDiff = res.bytesWritten - lastBytes;
            const speed = bytesDiff / timeDiff; // bytes/segundo
            
            setDownloadState({
              progress: res.bytesWritten / res.contentLength,
              speed,
              bytesWritten: res.bytesWritten,
              totalBytes: res.contentLength,
              status: 'downloading',
            });

            lastUpdate = now;
            lastBytes = res.bytesWritten;
          }
        },
        progressDivider: 1,
      });

      await job.promise;

      // Instalación simulada
      setDownloadState(prev => ({ ...prev, status: 'installing' }));
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Guardar ruta del modelo
      await saveModelPath(destPath);

      // Completado
      setDownloadState(prev => ({ ...prev, status: 'complete', progress: 1 }));
      
      // Navegar al chat
      setTimeout(() => {
        navigation.replace('Chat', { modelPath: destPath });
      }, 1000);

    } catch (error) {
      console.error('Download failed:', error);
      setDownloadState(prev => ({
        ...prev,
        status: 'error',
        error: 'Error al descargar el modelo. Por favor, intenta de nuevo.',
      }));
    }
  };

  const handleRetry = () => {
    setDownloadState({
      progress: 0,
      speed: 0,
      bytesWritten: 0,
      totalBytes: 0,
      status: 'idle',
    });
    checkAndDownloadModel();
  };

  const handleCancel = () => {
    // Aquí podrías implementar la lógica para cancelar la descarga
    // Por ahora, solo navegamos de vuelta
    navigation.goBack();
  };

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
  };

  const formatSpeed = (bytesPerSecond: number): string => {
    return `${formatBytes(bytesPerSecond)}/s`;
  };

  const getStatusText = (): string => {
    switch (downloadState.status) {
      case 'downloading':
        return 'Descargando...';
      case 'installing':
        return 'Instalando...';
      case 'complete':
        return '¡Listo!';
      case 'error':
        return 'Error';
      default:
        return 'Preparando...';
    }
  };

  const getTimeRemaining = (): string => {
    if (downloadState.speed === 0 || downloadState.progress === 0) {
      return '';
    }

    const remaining = downloadState.totalBytes - downloadState.bytesWritten;
    const seconds = Math.ceil(remaining / downloadState.speed);

    if (seconds < 60) {
      return `${seconds} seg restantes`;
    } else {
      const minutes = Math.ceil(seconds / 60);
      return `${minutes} min restantes`;
    }
  };

  if (downloadState.status === 'error') {
    return (
      <View style={styles.container}>
        <View style={styles.content}>
          <Text style={styles.title}>Error al descargar</Text>
          <Text style={styles.errorText}>{downloadState.error}</Text>
          <Button
            title="Reintentar"
            onPress={handleRetry}
            fullWidth
          />
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.title}>Preparando tu asistente personal</Text>

        <View style={styles.progressContainer}>
          <CircularProgress
            progress={downloadState.progress}
            size={220}
            strokeWidth={14}
            speed={downloadState.speed > 0 ? formatSpeed(downloadState.speed) : undefined}
          />
        </View>

        <View style={styles.infoContainer}>
          <Text style={styles.infoTitle}>
            Descargando modelo IA (1.3 GB)
          </Text>
          <Text style={styles.infoSubtitle}>
            Esto solo sucede una vez
          </Text>
          
          {downloadState.totalBytes > 0 && (
            <View style={styles.progressBar}>
              <View
                style={[
                  styles.progressBarFill,
                  { width: `${downloadState.progress * 100}%` },
                ]}
              />
            </View>
          )}

          <View style={styles.statsContainer}>
            <Text style={styles.statsText}>{getStatusText()}</Text>
            {downloadState.speed > 0 && (
              <Text style={styles.statsText}>{getTimeRemaining()}</Text>
            )}
          </View>
        </View>

        {downloadState.status === 'downloading' && (
          <Button
            title="Cancelar"
            onPress={handleCancel}
            variant="ghost"
            fullWidth
          />
        )}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: Spacing.xl,
  },
  title: {
    ...Typography.h2,
    color: Colors.text,
    textAlign: 'center',
    marginBottom: Spacing.xxl,
  },
  progressContainer: {
    marginBottom: Spacing.xxl,
  },
  infoContainer: {
    width: '100%',
    alignItems: 'center',
    marginBottom: Spacing.xl,
  },
  infoTitle: {
    ...Typography.body,
    color: Colors.text,
    textAlign: 'center',
    marginBottom: Spacing.xs,
  },
  infoSubtitle: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
    textAlign: 'center',
    marginBottom: Spacing.lg,
  },
  progressBar: {
    width: '100%',
    height: 4,
    backgroundColor: Colors.backgroundCard,
    borderRadius: 2,
    overflow: 'hidden',
    marginBottom: Spacing.md,
  },
  progressBarFill: {
    height: '100%',
    backgroundColor: Colors.primary,
  },
  statsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
  },
  statsText: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
  },
  errorText: {
    ...Typography.body,
    color: Colors.error,
    textAlign: 'center',
    marginBottom: Spacing.xl,
  },
});

export default DownloadScreen;

