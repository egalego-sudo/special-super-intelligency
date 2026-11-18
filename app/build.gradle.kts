plugins { id("com.android.application") }

android {
    namespace = "com.bittupandit.ssi"
    compileSdk = 35
    defaultConfig {
        applicationId = "com.bittupandit.ssi"
        minSdk = 23
        targetSdk = 35
        versionCode = 3
        versionName = "1.2"
    }
}

dependencies {
    testImplementation("junit:junit:4.13.2")
}
