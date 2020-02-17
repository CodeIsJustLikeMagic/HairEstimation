import ij.*;
import ij.process.*;
import ij.gui.*;
import java.awt.*;
import ij.plugin.*;
import ij.plugin.frame.*;
import java.lang.Math.*;

public class ex_01_3_jpfeif2sCMYKChannelSeperate implements PlugIn {

	public void run(String arg) {
		ImagePlus imp = IJ.getImage();
		splitPlus(imp.duplicate());
	}

	private void splitPlus(ImagePlus mImage){
		//create Duplicate Images for magenta and yellow
		ImagePlus cImage = mImage.duplicate();
		ImagePlus yImage = mImage.duplicate();
		cImage.setTitle("Cyan color Channel");
		mImage.setTitle("Magenta color Channel");
		yImage.setTitle("Yellow color Channel");
		
		//create ImageProcessors for all the things
		ImageProcessor mIp = mImage.getProcessor();
		ImageProcessor cIp = cImage.getProcessor();
		ImageProcessor yIp = yImage.getProcessor();
		
		int w = mIp.getWidth();
		int h = mIp.getHeight();
		int[] color = new int[3]; 
		int r,g,b, c,m,y,k;
		//create blank 8-bit grayscale image to display K
		ImagePlus kImage = IJ.createImage("K-Komponente", "8-bit grayscale",w,h,1);
		ImageProcessor kIp = kImage.getProcessor();
		
		//go through all the Pixels
		for(int u = 0; u<w;u++)
		{
			for(int v = 0; v<h;v++)
			 {
				//save colors in c
				color = mIp.getPixel(u,v,color);
				r = color[0];
				g = color[1];
				b = color[2];
				//convert to CMY (CMY) = (111)-(RGB)
				//1 is white. white light without the color red is cyan.
				//in imageJ 0 is black and 255 white. 255-red= cyan
				c=255-r;
				m=255-g;
				y=255-b; 
				
				//convert to CMYK
				//(K is the lightness value)
				//the gray value k is the darkest(smallest) value of cmy 
				k = Math.min(c, Math.min(m, y));
				//subtract the grayvalue from the cmy
				//the brightness doesn't have to come from the cmy channels since it comes from k
				c=(c-k);
				m=(m-k);
				y=(y-k);
				//this results in images over black background.	
				
				//assign colors to Channel Images
				//display CMYK over black background
				/*
				int[] cc = {0,c,c};
				int[] mm = {m,0,m}; 
				int[] yy = {y,y,0};
				*/
				//display CMY over white background. (all the calculations are unnessesary for this)
				//full red and green makes yellow. 
				//  if you add blue to that, the yellow becomes less intense.
				//  add full blue and you get white. add no blue and you get yellow.
				/*
				int[] cc = {r,255,255};
				int[] mm = {255,g,255};
				int[] yy = {255,255,b}; 
				*/
				//diplay CMYK over white background.
				//100% red and 100% green create 100% yellow.
				//255-y is blue so adding that will bring the color closer to white. 
				//The Color Spectrum is limited to everything between white and yellow.
				//           r   g    b
				int[] cc = {255-c,255,255};
				int[] mm = {255,255-m,255};
				int[] yy = {255,255,255-y}; 
				yIp.putPixel(u,v,yy);
				cIp.putPixel(u,v,cc);
				mIp.putPixel(u,v,mm);
				kIp.putPixel(u,v,255-k); //inverting k
			}
		}
		//display the channel images
		mImage.show();
		mImage.draw();
		cImage.show();
		cImage.draw();
		yImage.show();
		yImage.draw();
		kImage.show();
		kImage.draw();

	}


}
