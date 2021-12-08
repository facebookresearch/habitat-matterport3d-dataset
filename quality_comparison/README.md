# Scene quality comparison for HM3D
We measure the reconstruction quality of a 3D scene by measuring its reconstruction completeness and visual fidelity.

We first render RGB-D images from the scene using Habitat. To measure reconstruction completeness, we define a view-based metric that measures the degree to which reconstruction artifacts (or defects) occur in an image. Reconstruction artifacts include missing surfaces, holes, and untextured surface regions. We then quantify the fraction of viewpoints in the scene that have a significant number of defects (i.e., the %defects metric).
To measure visual fidelity, we compare the rendered RGB images with real-world captured images generated from high-resolution panoramas in Gibson and MP3D using divergence metrics such as KID and FID.

We now provide instructions to compute the reconstruction completeness and visual fidelity metrics in the [HM3D paper](https://openreview.net/pdf?id=-v4OuqNs5P) (Fig. 4 and Table 5, respectively).


1. **Simulated image extraction:** Ensure that the environment variables corresponding to the different datasets are set as instructed in the main README. We render simulated images for each dataset by sampling locations in a uniform grid of navigable locations. Set the environment variable `SAVE_DIR_PATH` to save the extracted images and run the extraction script.

    ```
    export SAVE_DIR_PATH="< directory to save extracted images >"
    chmod +x run_sim_extraction.sh && ./run_sim_extraction.sh
    ```

2. **Real-world image extraction:** We extract real-world images by randomly sampling perspective images from the raw 360 panoramas released in Gibson and MP3D.

    a. **Obtain 360 panoramas for MP3D:** Download the [PanoBasic toolkit](https://github.com/yindaz/PanoBasic/blob/master/demo_matterport.m) to some path (say, `PANO_BASIC_ROOT`). Copy all the MP3D skybox images to a single directory (say, `MP3D_SKYBOX_ROOT`). See [MP3D dataset organization](https://github.com/niessner/Matterport/blob/master/data_organization.md) for reference. Let the path to save the MP3D panoramas be `MP3D_PANO_SAVE_ROOT`. Update these root paths in `convert_mp3d_to_pano.m` from this directory. Run `convert_mp3d_to_pano.m` using matlab to obtain the MP3D panoramas.

    b. **Extract images from panoramas:** Set the panorama paths as environment variables and run the extraction script.
    ```
    export GIBSON_PANO_ROOT="< path to gibson raw dataset >"
    export MP3D_PANO_ROOT="< path to mp3d panoramas >"
    chmod +x run_real_extraction.sh && ./run_real_extraction.sh
    ```

3. **Measuring visual fidelity:** Execute the visual fidelity comparison script.
    ```
    chmod +x run_visual_fidelity.sh && ./run_visual_fidelity.sh
    ```

4. **Measuring reconstruction completeness:** Execute the reconstruction completeness comparison script.
    ```
    chmod +x run_reconstruction_completeness.sh && ./run_reconstruction_completeness.sh
    ```
