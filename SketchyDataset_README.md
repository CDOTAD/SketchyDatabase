Sketchy Database - Rendered sketches and augmented photos

Contents:
  photo - a directory containing two different renderings of 
    all photographs contained within the Sketchy Database
  sketch - a directory containing six different renderings of
    all sketches contained within the Sketchy Database

Photographs
    All photographs are rendered in JPEG format. Resizing is
  performed via OpenCV Imgproc (typically area interpolation
  for full images renderings and cubic for bounding box
  renderings).

  Augmentations (directories within 'photo')
  
    tx_000000000000 : image is non-uniformly scaled to 256x256
    tx_000100000000 : image bounding box scaled to 256x256 with
                      an additional +10% on each edge; note 
                      that due to position within the image,
                      sometimes the object is not centered

Sketches
    All sketches are rendered in PNG format. The original
  sketch canvas size is 640x480. In rendering the sketch to a
  256x256 canvas, we take the original photo aspect ratio
  as well as the original sketch canvas aspect ratio into
  account. We render sketches such that they are consistent
  with the transformation made to the image (non-uniform
  scale to 256x256). In order to ensure sketches remain fully
  on the canvas, some minor adjustments to scale and/or 
  location are occasionally necessary.
    All sketches are rendered using custom OpenGL code, with
  a PNG encoding provided by Java's ImageIO API.

  Augmentations (directories within 'sketch')

    tx_000000000000 : sketch canvas is rendered to 256x256
                      such that it undergoes the same
                      scaling as the paired photo
    tx_000100000000 : sketch is centered and uniformly scaled 
                      such that its greatest dimension (x or y) 
                      fills 78% of the canvas (roughly the same
                      as in Eitz 2012 sketch data set)
    tx_000000000010 : sketch is translated such that it is 
                      centered on the object bounding box
    tx_000000000110 : sketch is centered on bounding box and
                      is uniformly scaled such that one dimension
                      (x or y; whichever requires the least amount
                      of scaling) fits within the bounding box
    tx_000000001010 : sketch is centered on bounding box and
                      is uniformly scaled such that one dimension
                      (x or y; whichever requires the most amount
                      of scaling) fits within the bounding box
    tx_000000001110 : sketch is centered on bounding box and
                      is non-uniformly scaled such that it 
                      completely fits within the bounding box


