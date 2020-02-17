import ij.*;
import ij.process.*;
import ij.gui.*;
import java.awt.*;
import ij.plugin.*;
import ij.plugin.frame.*;
import java.lang.Math;
public class invert implements PlugIn {
//input is a grayscale image
//output is binary grayscale image
//using local adaptive threshhold

	public void run(String arg) {
		ImagePlus imp = IJ.getImage();
		ImagePlus resImage = FixedThreshhold(imp.duplicate(),140);
	}
	private ImagePlus FixedThreshhold(ImagePlus imp,int threshhold) {
		//transform into a binary image
		ImageProcessor ip = imp.getProcessor();
		int w = ip.getWidth();
		int h = ip.getHeight();
		ImagePlus resultImage = IJ.createImage("Inverted", "8-Bit", w, h, 1);
		ImageProcessor resultIp = resultImage.getProcessor();

		int[] c = new int[3];
		for (int u = 0; u < w; u++) {
			for (int v = 0; v < h; v++) {
				//make the class binary according to the threshold t
				c = ip.getPixel(u, v, c);
				resultIp.putPixel(u, v, 255 - c[0]);
			}
		}
			resultImage.show();
			resultImage.draw();
			return resultImage;
	}

}
