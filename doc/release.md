# Merging a PR to Master Branch
- **IMPORTANT** When we change APIs, make sure to increment the minor version number so we can distinguish one version to another. It'll change the project name and hiprt DLL name too. 
- When we don't change APIs, no need to increment the version number. However, each PR should update the patch number so we can find the source of the binary. 

# Release Procedure
- Update hiprt package in the **internal** [hiprtSdk repo](https://github.com/amdadvtech/hiprtsdk)
  - A package is built by [HIPRT_Release](http://arrtc.amd.com:8111/viewType.html?buildTypeId=Firerender_HiprtRelease) TC project. It gathers artifacts and push to artifactory. The build number is used for the name of the zip (hiprtSdkCore_7.zip for example). 
- Confirm tutorial works
- Confirm HiprtEW path works
- Preparing Zip for release 
  - Freshly clone [hiprtSdk repo](https://github.com/amdadvtech/hiprtsdk).
  - Fetch submodules 
  - Zip hiprtSdk folder
- Update hiprt package at [gpuopen](https://gpuopen.com/hiprt/). 
  - Once we prepared the zip, talk to GPUOpen web team ask them to update the page and zip (rosanna.ashworth-jones@amd.com, thomas.lewis@amd.com).
- Create a new release and update the release branch (e.g., `release/hiprt2.x`) in the hiprt **internal** repo
- Update the hiprt tutorial **public** repo and create a new release

# HIPRT components for Driver
- Driver team is looking at artifactory http://scvartifactory.amd.com/ui/repos/tree/General/HIPRT/
  - Here, it should contain the kernel library which user do not link against their code, i.e., BVH build kernel etc.
- The package should be updated by [Deploy To Artifactory](http://arrtc.amd.com:8111/viewType.html?buildTypeId=Firerender_Hiprt_DeployToArtifactory) TC project
 
