import ij.*;
import ij.process.*;
import ij.gui.*;
import java.awt.*;
import ij.plugin.*;
import ij.plugin.frame.*;
import java.lang.Math;
public class HairDetection implements PlugIn {
//input is a grayscale image
//output is binary grayscale image
//using local adaptive threshhold

	public void run(String arg) {
		ImagePlus imp = IJ.getImage();
		ImagePlus resImage = FixedThreshhold(imp.duplicate(),140);
	}

	private void grayscale (ImagePlus imp){
		ImageProcessor ip = imp.getProcessor();
		int w = ip.getWidth();
		int h = ip.getHeight();
		ImagePlus lumImage = IJ.createImage("luminanz grayscale", "8-bit grayscale",w,h,1);
		ImageProcessor lumIp = lumImage.getProcessor();

		int[] c = new int[3];
		for(int u = 0; u<w;u++)
		{
			for(int v = 0; v<h;v++)
			{
				c = ip.getPixel(u,v,c);
				//grayscale with luminanz
				int lumval =(int) ((0.3*(double)c[0])+(0.6*(double)c[1])+(0.1*(double)c[2]));
				int[] lum = {(int)lumval};
				lumIp.putPixel(u,v,lum);
			}
		}
		lumImage.show();
		lumImage.draw();
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
