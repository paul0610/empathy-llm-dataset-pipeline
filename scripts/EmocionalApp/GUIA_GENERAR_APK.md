# Guía Completa: Generar APK de Producción

## Fecha: Octubre 12, 2025

---

## RESUMEN

Esta guía describe el proceso completo para generar un APK de producción de la aplicación Emocional, desde el bundle de JavaScript hasta la firma y generación del APK final.

---

## PREREQUISITOS

Antes de comenzar, asegúrate de tener:

- ✅ **Node.js** instalado (versión 18 o superior)
- ✅ **Android Studio** instalado
- ✅ **JDK** (Java Development Kit) instalado
- ✅ **Android SDK** configurado
- ✅ Proyecto React Native funcionando correctamente en modo desarrollo

---

## PASO 1: GENERAR EL BUNDLE DE JAVASCRIPT

El primer paso es generar el bundle de JavaScript que contendrá todo el código de tu aplicación.

### Opción A: Usando el script npm (Recomendado)

```bash
cd /ruta/a/tu/proyecto/app_emotional_improved
npm run bundle-android
```

### Opción B: Comando manual

```bash
npx react-native bundle \
  --platform android \
  --dev false \
  --entry-file index.js \
  --bundle-output android/app/src/main/assets/index.android.bundle \
  --assets-dest android/app/src/main/res
```

### ¿Qué hace este comando?

- `--platform android`: Especifica que es para Android
- `--dev false`: Modo producción (sin debugging)
- `--entry-file index.js`: Archivo de entrada de la app
- `--bundle-output`: Donde se guardará el bundle generado
- `--assets-dest`: Donde se guardarán los assets (imágenes, fuentes, etc.)

### Verificar que se creó el bundle

```bash
ls -lh android/app/src/main/assets/index.android.bundle
```

Deberías ver un archivo con tamaño considerable (varios MB).

---

## PASO 2: LIMPIAR PROYECTO (OPCIONAL PERO RECOMENDADO)

Antes de generar el APK, es buena práctica limpiar el proyecto:

```bash
cd android
./gradlew clean
cd ..
```

---

## PASO 3: GENERAR EL APK

Ahora puedes generar el APK de dos formas:

### Opción A: APK de Debug (para pruebas)

```bash
cd android
./gradlew assembleDebug
cd ..
```

El APK se generará en:
```
android/app/build/outputs/apk/debug/app-debug.apk
```

### Opción B: APK de Release (para producción)

```bash
cd android
./gradlew assembleRelease
cd ..
```

El APK se generará en:
```
android/app/build/outputs/apk/release/app-release.apk
```

**IMPORTANTE**: Para generar el APK de release, necesitas configurar la firma (ver Paso 4).

---

## PASO 4: CONFIGURAR FIRMA DEL APK (SOLO PARA RELEASE)

Para publicar la app en Google Play Store o distribuirla, necesitas firmar el APK.

### 4.1. Generar una Keystore

Si aún no tienes una keystore, genera una:

```bash
cd android/app
keytool -genkeypair -v -storetype PKCS12 -keystore emocional-release-key.keystore -alias emocional-key-alias -keyalg RSA -keysize 2048 -validity 10000
```

Te pedirá:
- **Contraseña de la keystore**: Elige una contraseña segura y guárdala
- **Nombre y apellido**: Tu nombre o el de la organización
- **Unidad organizativa**: Opcional
- **Organización**: Nombre de tu empresa/proyecto
- **Ciudad, Estado, País**: Tu ubicación

**⚠️ IMPORTANTE**: Guarda la keystore y la contraseña en un lugar seguro. Si las pierdes, no podrás actualizar la app en Google Play Store.

### 4.2. Configurar gradle.properties

Edita el archivo `android/gradle.properties` y añade al final:

```properties
MYAPP_RELEASE_STORE_FILE=emocional-release-key.keystore
MYAPP_RELEASE_KEY_ALIAS=emocional-key-alias
MYAPP_RELEASE_STORE_PASSWORD=tu_contraseña_keystore
MYAPP_RELEASE_KEY_PASSWORD=tu_contraseña_keystore
```

**⚠️ SEGURIDAD**: No subas este archivo a Git con las contraseñas. Usa variables de entorno o archivos locales.

### 4.3. Configurar build.gradle

Edita el archivo `android/app/build.gradle` y añade la configuración de firma:

```gradle
android {
    ...
    defaultConfig { ... }
    
    signingConfigs {
        release {
            if (project.hasProperty('MYAPP_RELEASE_STORE_FILE')) {
                storeFile file(MYAPP_RELEASE_STORE_FILE)
                storePassword MYAPP_RELEASE_STORE_PASSWORD
                keyAlias MYAPP_RELEASE_KEY_ALIAS
                keyPassword MYAPP_RELEASE_KEY_PASSWORD
            }
        }
    }
    
    buildTypes {
        release {
            ...
            signingConfig signingConfigs.release
        }
    }
}
```

### 4.4. Generar APK firmado

Ahora sí, genera el APK de release firmado:

```bash
cd android
./gradlew assembleRelease
cd ..
```

---

## PASO 5: USAR ANDROID STUDIO (ALTERNATIVA)

También puedes usar Android Studio para generar el APK:

### 5.1. Abrir el proyecto en Android Studio

1. Abre Android Studio
2. Selecciona "Open an Existing Project"
3. Navega a la carpeta `android` dentro de tu proyecto
4. Haz clic en "OK"

### 5.2. Generar el APK desde Android Studio

1. En el menú superior, selecciona **Build → Build Bundle(s) / APK(s) → Build APK(s)**
2. Espera a que se complete el proceso
3. Aparecerá una notificación con un enlace "locate" para encontrar el APK

### 5.3. Generar APK firmado desde Android Studio

1. En el menú superior, selecciona **Build → Generate Signed Bundle / APK**
2. Selecciona **APK** y haz clic en **Next**
3. Si ya tienes una keystore:
   - Haz clic en **Choose existing...**
   - Selecciona tu archivo `.keystore`
   - Ingresa la contraseña
4. Si no tienes keystore:
   - Haz clic en **Create new...**
   - Completa los campos
   - Guarda la keystore en un lugar seguro
5. Selecciona **release** como Build Variant
6. Marca las opciones:
   - ✅ V1 (Jar Signature)
   - ✅ V2 (Full APK Signature)
7. Haz clic en **Finish**

El APK firmado se generará en `android/app/release/app-release.apk`

---

## PASO 6: OPTIMIZAR EL APK (OPCIONAL)

### 6.1. Habilitar ProGuard

ProGuard reduce el tamaño del APK y ofusca el código. Edita `android/app/build.gradle`:

```gradle
buildTypes {
    release {
        minifyEnabled true
        shrinkResources true
        proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        signingConfig signingConfigs.release
    }
}
```

### 6.2. Habilitar splits por ABI

Para generar APKs más pequeños específicos para cada arquitectura:

```gradle
android {
    ...
    splits {
        abi {
            reset()
            enable true
            universalApk false
            include "armeabi-v7a", "arm64-v8a", "x86", "x86_64"
        }
    }
}
```

Esto generará múltiples APKs (uno por arquitectura) en lugar de uno universal.

---

## PASO 7: VERIFICAR EL APK

### 7.1. Verificar la firma

```bash
jarsigner -verify -verbose -certs android/app/build/outputs/apk/release/app-release.apk
```

Deberías ver: `jar verified.`

### 7.2. Ver información del APK

```bash
aapt dump badging android/app/build/outputs/apk/release/app-release.apk
```

### 7.3. Instalar el APK en un dispositivo

```bash
adb install android/app/build/outputs/apk/release/app-release.apk
```

O simplemente copia el APK a tu dispositivo y ábrelo para instalarlo.

---

## PASO 8: GENERAR AAB (ANDROID APP BUNDLE) PARA GOOGLE PLAY

Google Play Store recomienda usar AAB en lugar de APK:

```bash
cd android
./gradlew bundleRelease
cd ..
```

El AAB se generará en:
```
android/app/build/outputs/bundle/release/app-release.aab
```

---

## RESUMEN DE COMANDOS

### Flujo completo para generar APK de release:

```bash
# 1. Generar bundle de JavaScript
npm run bundle-android

# 2. Limpiar proyecto
cd android
./gradlew clean

# 3. Generar APK de release
./gradlew assembleRelease
cd ..

# 4. El APK estará en:
# android/app/build/outputs/apk/release/app-release.apk
```

### Flujo completo para generar AAB (Google Play):

```bash
# 1. Generar bundle de JavaScript
npm run bundle-android

# 2. Limpiar proyecto
cd android
./gradlew clean

# 3. Generar AAB de release
./gradlew bundleRelease
cd ..

# 4. El AAB estará en:
# android/app/build/outputs/bundle/release/app-release.aab
```

---

## SOLUCIÓN DE PROBLEMAS

### Error: "SDK location not found"

**Solución**: Crea el archivo `android/local.properties` con:

```properties
sdk.dir=/ruta/a/tu/Android/Sdk
```

En Windows:
```properties
sdk.dir=C:\\Users\\TuUsuario\\AppData\\Local\\Android\\Sdk
```

En macOS/Linux:
```properties
sdk.dir=/Users/TuUsuario/Library/Android/sdk
```

### Error: "Failed to install the app"

**Solución**: Desinstala la versión anterior primero:

```bash
adb uninstall com.emotionalapp
adb install android/app/build/outputs/apk/release/app-release.apk
```

### Error: "Execution failed for task ':app:mergeReleaseResources'"

**Solución**: Limpia los recursos duplicados:

```bash
cd android
./gradlew clean
cd ..
rm -rf android/app/src/main/res/drawable-*
npm run bundle-android
cd android
./gradlew assembleRelease
```

### APK muy grande

**Solución**: 
1. Habilita ProGuard (ver Paso 6.1)
2. Habilita splits por ABI (ver Paso 6.2)
3. Usa AAB en lugar de APK (Google Play lo optimizará automáticamente)

---

## CHECKLIST FINAL

Antes de distribuir el APK, verifica:

- [ ] El bundle de JavaScript está actualizado
- [ ] El APK está firmado correctamente
- [ ] La versión en `android/app/build.gradle` está actualizada
- [ ] Has probado el APK en un dispositivo real
- [ ] El tamaño del APK es razonable (< 100 MB idealmente)
- [ ] Todas las funcionalidades funcionan correctamente
- [ ] No hay logs de debugging en producción
- [ ] Los permisos en `AndroidManifest.xml` son correctos

---

## NOTAS ADICIONALES

### Actualizar versión de la app

Edita `android/app/build.gradle`:

```gradle
android {
    defaultConfig {
        ...
        versionCode 2          // Incrementa este número
        versionName "1.1.0"    // Actualiza la versión visible
    }
}
```

### Permisos necesarios

Verifica que `android/app/src/main/AndroidManifest.xml` tenga los permisos necesarios:

```xml
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
```

---

**Autor**: Manus AI  
**Fecha**: Octubre 12, 2025

