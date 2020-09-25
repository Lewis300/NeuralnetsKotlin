plugins {
    java
    kotlin("jvm") version "1.3.72"
}

group = "com.lewispanos"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
    flatDir {
        dirs("libs")
    }
}

dependencies {
    implementation(kotlin("stdlib-jdk8"))
    testCompile("junit", "junit", "4.12")
    //gradle kotlin DSL
    implementation("com.github.doyaaaaaken:kotlin-csv-jvm:0.10.4")

    implementation ("no.tornado:tornadofx:1.7.20")

    implementation(fileTree("libs"))
}

configure<JavaPluginConvention> {
    sourceCompatibility = JavaVersion.VERSION_1_8
}
tasks {
    compileKotlin {
        kotlinOptions.jvmTarget = "1.8"
    }
    compileTestKotlin {
        kotlinOptions.jvmTarget = "1.8"
    }
}