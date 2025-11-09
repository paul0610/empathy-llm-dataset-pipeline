# Aplicación Emocional - Proyecto Completo

## 📦 Contenido del Proyecto

Este es un proyecto **React Native 0.78.1 completo** con todas las carpetas nativas (`android/` e `ios/`) y el código mejorado basado en los prototipos UX de Stitch.

---

## ✅ Lo que Incluye

- ✅ **Carpeta `android/` completa** con Gradle, AndroidManifest.xml, etc.
- ✅ **Carpeta `ios/` completa** (si necesitas compilar para iOS)
- ✅ **Código fuente mejorado** (4 pantallas: Onboarding, Descarga, Chat, Configuración)
- ✅ **Componentes reutilizables** (Button, Input, CircularProgress, ChatBubble)
- ✅ **Sistema de diseño profesional** basado en prototipos de Stitch
- ✅ **Dependencias instaladas** (975 paquetes)
- ✅ **Configuración completa** (metro.config.js, babel.config.js, tsconfig.json)

---

## 🚀 Pasos para Ejecutar la Aplicación

### 1. Extraer el Proyecto

Extrae el archivo ZIP en tu directorio de proyectos:

```
D:\proyectos\LLMS\TFM\EmocionalApp\
```

### 2. Verificar Dependencias

Las dependencias ya están instaladas, pero si necesitas reinstalarlas:

```bash
cd EmocionalApp
npm install
```

### 3. Configurar Android SDK

Asegúrate de tener configurado el Android SDK. Crea el archivo `android/local.properties`:

```properties
sdk.dir=C:\\Users\\TuUsuario\\AppData\\Local\\Android\\Sdk
```

### 4. Iniciar Metro Bundler

En una terminal:

```bash
npm start
```

### 5. Ejecutar en Android

En otra terminal:

```bash
npx react-native run-android
```

O desde Android Studio:
1. Abre la carpeta `android/` en Android Studio
2. Espera a que Gradle sincronice
3. Haz clic en el botón "Run" (▶)

---

## 📱 Generar APK

### Opción 1: Desde la Terminal

```bash
# 1. Generar bundle de JavaScript
npm run bundle-android

# 2. Generar APK de debug
cd android
./gradlew assembleDebug
cd ..

# El APK estará en:
# android/app/build/outputs/apk/debug/app-debug.apk
```

### Opción 2: Desde Android Studio

1. Abre `android/` en Android Studio
2. **Build → Build Bundle(s) / APK(s) → Build APK(s)**
3. Espera a que se complete
4. Haz clic en "locate" para encontrar el APK

---

## 🔧 Configuración de Dependencias Nativas

Las siguientes dependencias **requieren configuración nativa** (autolinking debería hacerlo automáticamente):

### AsyncStorage
```bash
npm install @react-native-async-storage/async-storage
```

### React Navigation
```bash
npm install @react-navigation/native @react-navigation/stack
npm install react-native-screens react-native-safe-area-context
npm install react-native-gesture-handler react-native-reanimated
```

### SVG
```bash
npm install react-native-svg
```

### File System
```bash
npm install react-native-fs
```

### Llama.rn (Modelo de IA)
```bash
npm install llama.rn
```

**NOTA**: React Native 0.78.1 tiene **autolinking**, por lo que estas dependencias deberían configurarse automáticamente. Si tienes problemas, ejecuta:

```bash
cd android
./gradlew clean
cd ..
npx react-native run-android
```

---

## 📂 Estructura del Proyecto

```
EmocionalApp/
├── android/                    # Carpeta nativa de Android
│   ├── app/
│   │   ├── build.gradle       # Configuración de la app
│   │   └── src/main/
│   │       ├── AndroidManifest.xml
│   │       ├── java/
│   │       └── res/
│   ├── build.gradle           # Configuración de Gradle
│   ├── gradle/
│   ├── gradlew                # Script de Gradle (Linux/Mac)
│   ├── gradlew.bat            # Script de Gradle (Windows)
│   └── settings.gradle
├── ios/                        # Carpeta nativa de iOS
├── src/                        # Código fuente de la app
│   ├── components/            # Componentes reutilizables
│   │   ├── Button.tsx
│   │   ├── ChatBubble.tsx
│   │   ├── CircularProgress.tsx
│   │   └── Input.tsx
│   ├── constants/
│   │   └── DesignSystem.ts    # Sistema de diseño
│   ├── screens/               # Pantallas principales
│   │   ├── OnboardingScreen.tsx
│   │   ├── DownloadScreen.tsx
│   │   ├── ChatScreen.tsx
│   │   └── SettingsScreen.tsx
│   ├── types/
│   │   └── index.ts           # Tipos TypeScript
│   └── utils/
│       └── storage.ts         # Utilidades de AsyncStorage
├── App.tsx                     # Componente principal
├── index.js                    # Punto de entrada
├── package.json                # Dependencias
├── metro.config.js             # Configuración de Metro
├── babel.config.js             # Configuración de Babel
└── tsconfig.json               # Configuración de TypeScript
```

---

## 🎨 Características Implementadas

### Pantallas

1. **Onboarding** (Nueva)
   - Wizard de 3 pasos
   - Bienvenida, términos y contacto de emergencia
   
2. **Descarga** (Mejorada)
   - Progreso circular animado
   - Velocidad y tiempo restante
   
3. **Chat** (Rediseñada)
   - Header con estado online
   - Burbujas con avatares
   - Indicador de "escribiendo..."
   
4. **Configuración** (Nueva)
   - Gestión de contacto de emergencia
   - Estadísticas de conversaciones
   - Privacidad y seguridad

### Componentes Reutilizables

- **Button**: Variantes (primary, secondary, danger, ghost)
- **Input**: Con label y manejo de errores
- **CircularProgress**: Progreso circular animado
- **ChatBubble**: Burbujas de chat diferenciadas

---

## ⚠️ Problemas Comunes

### Error: "SDK location not found"

Crea el archivo `android/local.properties`:

```properties
sdk.dir=C:\\Users\\TuUsuario\\AppData\\Local\\Android\\Sdk
```

### Error: "Failed to install the app"

Desinstala la versión anterior:

```bash
adb uninstall com.emotionalapp
npx react-native run-android
```

### Error: Metro Bundler no inicia

Limpia la caché:

```bash
npm start -- --reset-cache
```

### Error de compilación de Gradle

Limpia el proyecto:

```bash
cd android
./gradlew clean
cd ..
npx react-native run-android
```

---

## 📝 Próximos Pasos

1. ✅ **Ejecuta la app** para verificar que todo funciona
2. ✅ **Configura llama.rn** con tu modelo de IA
3. ✅ **Personaliza** colores y textos según tus necesidades
4. ✅ **Genera el APK** para distribuir

---

## 🔗 Recursos

- [Documentación de React Native](https://reactnative.dev/)
- [React Navigation](https://reactnavigation.org/)
- [Llama.rn](https://github.com/mybigday/llama.rn)

---

## 📧 Soporte

Si tienes problemas, revisa:
1. La guía `GUIA_GENERAR_APK.md` para compilar el APK
2. La documentación oficial de React Native
3. Los logs de error en la terminal

---

**Proyecto generado**: Octubre 12, 2025  
**React Native**: 0.78.1  
**Node.js**: 22.13.0

