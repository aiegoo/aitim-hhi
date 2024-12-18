# PowerShell script to start the jetson-inference docker container for x86_64 architecture

param(
    [string]$ContainerImage = "", # Docker container image name (default loaded from tag logic)
    [string]$RosDistro = "",      # ROS distribution (optional)
    [string]$RunCommand = "",    # Command to execute inside the container
    [string[]]$Volumes = @(),      # List of host:container volume mappings
    [switch]$DevMode,              # Development mode switch
    [switch]$Help                  # Show help and exit
)

function Show-Help {
    Write-Host "\nUsage: Starts the Docker container and runs a user-specified command\n"
    Write-Host "    ./run.ps1 [-ContainerImage DOCKER_IMAGE] [-RosDistro ROS_DISTRO]"
    Write-Host "               [-RunCommand RUN_COMMAND] [-Volumes HOST_DIR:MOUNT_DIR] [-DevMode]"
    Write-Host "Options:"
    Write-Host "    -ContainerImage   Specifies the name of the Docker container image"
    Write-Host "    -RosDistro        Specifies the ROS distro to use (e.g., 'noetic', 'foxy')"
    Write-Host "    -RunCommand       Command to run inside the container"
    Write-Host "    -Volumes          List of host:container volume mappings"
    Write-Host "    -DevMode          Enables development mode (mounts the project directory into the container)"
    Write-Host "    -Help             Displays this help message"
    exit
}

if ($Help) {
    Show-Help
}

# Default directories and paths
$ProjectDir = (Get-Location).Path
$DockerRoot = "/jetson-inference" # Path inside the Docker container

# Default volumes
$DataVolumes = @(
    "$ProjectDir\data:$DockerRoot/data",
    "$ProjectDir\python\training\classification\data:$DockerRoot/python/training/classification/data",
    "$ProjectDir\python\training\classification\models:$DockerRoot/python/training/classification/models",
    "$ProjectDir\python\training\detection\ssd\data:$DockerRoot/python/training/detection/ssd/data",
    "$ProjectDir\python\training\detection\ssd\models:$DockerRoot/python/training/detection/ssd/models",
    "$ProjectDir\python\www\recognizer\data:$DockerRoot/python/www/recognizer/data"
)

if ($DevMode) {
    $DataVolumes += "${ProjectDir}:${DockerRoot}"
}

# Add user-specified volumes
foreach ($Volume in $Volumes) {
    $DataVolumes += $Volume
}

# Get the container tag for x86_64
$ContainerTag = "22.06"  # Default tag for NGC pytorch base container

# Determine the container image based on ROS_DISTRO
if ($RosDistro) {
    $ContainerImage = "ros:$RosDistro-pytorch-l4t-$ContainerTag"
} else {
    $ContainerImage = "jetson-inference:$ContainerTag"
}

# Determine local and remote container images
$ContainerLocalImage = $ContainerImage
$ContainerRemoteImage = "dustynv/$ContainerLocalImage"

# Check if the local container image exists; otherwise, use the remote image
if (-not (docker images -q $ContainerLocalImage)) {
    $ContainerImage = $ContainerRemoteImage
}

# Build Docker run command
$VolumeArgs = ($DataVolumes | ForEach-Object { "--volume $_" }) -join " "
$DockerCommand = @(
    "docker run --rm -it",
    "--gpus all",
    "--network=host",
    "--shm-size=8g",
    "--ulimit memlock=-1",
    "--ulimit stack=67108864",
    "-e NVIDIA_DRIVER_CAPABILITIES=all",
    $VolumeArgs,
    "-w $DockerRoot",
    $ContainerImage,
    $RunCommand
) -join " "

# Display configuration for verification
Write-Host "Running Docker container with the following configuration:`n"
Write-Host "Container Image: $ContainerImage"
Write-Host "ROS Distro: $RosDistro"
Write-Host "Data Volumes: $($DataVolumes -join ", ")"
Write-Host "Development Mode: $($DevMode.IsPresent)"
Write-Host "Run Command: $RunCommand"
Write-Host "\nExecuting command:\n$DockerCommand\n"

# Execute the Docker command
Invoke-Expression $DockerCommand
