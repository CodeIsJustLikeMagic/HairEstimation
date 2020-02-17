import ij.*;
import ij.process.*;
import ij.gui.*;
import java.awt.*;
import ij.plugin.*;
import ij.plugin.frame.*;

public class ex_05_3_jpfeif2sSkelettierungZhanAndSuen implements PlugIn {
//Skelettierung Zhan and Suen
	public void run(String arg) {
		ImagePlus imp = IJ.getImage();
		//IJ.save(imp, "ex_05_3_Vorher.png");
		ImagePlus res = zhanAndSuenSkelett(imp);
		//IJ.save(res, res.getTitle());
		res = sub(imp,res, "orig and res",false);
		//IJ.save(res, res.getTitle());
		
	}
	private ImagePlus zhanAndSuenSkelett(ImagePlus imp) {
		return zhangAndSuenSkelett(imp, 1);
	}
	
	private ImagePlus zhangAndSuenSkelett(ImagePlus imp, int loopnum) {
		//input is a binary grayscale image
		boolean same = true;

		ImageProcessor ip = imp.getProcessor();
        
		int w = ip.getWidth();
		int h = ip.getHeight();
		ImagePlus resultImage = imp.duplicate();
		ImageProcessor resultIp = resultImage.getProcessor();

		int[] c = new int[3]; 
		int[] p = new int[11];
		int bp1=0;
		int ap1=0;
		//first subiteration
		for(int u = 1; u<w-1;u++)
		{
			for(int v = 1; v<h-1;v++)
			 {
				if(ip.getPixel(u,v,c)[0] == 0) {
					//retrieve p1-p9, A(p1) and B(p1)
					p = retrievePixelsNStuff(u, v, ip, c);
					bp1 = p[10];
					ap1 = p[11];
					/*
					for(int i = 0; i<p.length;i++) {
						IJ.log("p["+i+"] = "+p[i]);
					}
					*/
					//check for the conditions of the first sub iteration
					if(2<=bp1 && bp1 <=6 &&  ap1 == 1 &&  p[2-1]*p[4-1]*p[6-1]==0 && p[4-1]*p[6-1]*p[8-1]==0) {
						resultIp.putPixel(u,v,255);
						//IJ.log("first subiteration");
						same = false;
					}
				}
			 }
		}
		ImagePlus resultImage2 = resultImage.duplicate();
		resultImage2.setTitle("ex_05_3_ZahngAndSuen"+loopnum+".png");
		ImageProcessor resultIp2 = resultImage2.getProcessor();
		
		//second subiteration
		for(int u = w-2; u>0;u--)
		{
			for(int v = h-2; v>0 ;v--)
			 {
				if(ip.getPixel(u,v,c)[0] == 0) {
					//retrieve p1-p9, A(p1) and B(p1)
					p = retrievePixelsNStuff(u, v, resultIp, c);
					bp1 = p[10];
					ap1 = p[11];
					
					//check for the conditions of the second sub iteration
					if(2<=bp1 && bp1 <=6 &&  ap1 == 1 &&  p[2-1]*p[4-1]*p[8-1]==0 && p[2-1]*p[6-1]*p[8-1]==0) {
						resultIp2.putPixel(u,v,255);
						same = false;
						//IJ.log("second subiteration");
					}
				}
				
			 }
		}
		
		//check for differences between original image and resultImage2 (after second Subiteration)
		if(!same) {
			return zhangAndSuenSkelett(resultImage2,++loopnum);
		}
		resultImage2.show();
		resultImage2.draw();
		return resultImage2;
	}
	
	private int[] retrievePixelsNStuff(int u, int v, ImageProcessor ip, int[] c) {
		//retrieve Pixels under the structure clockwise starting in the middle.
		
		int[] p = new int[12];
		int prev = 0;
		int current = 0;
		//getPixel(row, line, c);
		p[1-1]= ip.getPixel(u,v,c)[0];
		p[2-1]=ip.getPixel(u,(v-1),c)[0];
		p[3-1]=ip.getPixel((u+1),(v-1),c)[0];
		p[4-1]=ip.getPixel((u+1),(v),c)[0];
		p[5-1]=ip.getPixel((u+1),(v+1),c)[0];
		p[6-1]=ip.getPixel((u),(v+1),c)[0];
		p[7-1]=ip.getPixel((u-1),(v+1),c)[0];
		p[8-1]=ip.getPixel((u-1),(v),c)[0];
		p[9-1]=ip.getPixel((u-1),(v-1),c)[0];
		p[10-1] = p[2-1];
		
		//convert grayscale values to theoretical 0 1 metric 0 is white. 1 is black
		for(int i =0; i<p.length;i++) {
			if(p[i] == 0) {//pixel is black (set)
				p[i]=1;
			}else { //pixel is white. not set
				p[i] = 0; 
			}
		}
		
		//number of set pixels under the structure p2-p9
		int bp1 =0;
		for(int i =1; i<p.length-3;i++) {
			if(p[i]==1) {
				bp1++;
			}
		}
		//number of 0->1 Changes from p2 to p9 clockwise
		int ap1 = 0;
		for(int i = 2; i<p.length-2;i++) {
			prev = p[i-1];
			current = p[i];
			if(prev == 0 && current == 1) {
				ap1++;
			}
		}
		
		p[10] = bp1;
		p[11] = ap1;
		return p;
		
	}
	
	
	
	private ImagePlus sub(ImagePlus original, ImagePlus lines, String designation,boolean save) {
		ImagePlus[] l = {lines};
		return sub( original,  l, designation,save);
 	}
	private ImagePlus sub(ImagePlus original, ImagePlus[] lines, String designation,boolean save) {
		//imp-imp2 +offset
		//input are two Grayscale Images
		ImageProcessor oIP = original.getProcessor();
		ImageProcessor[] linesIP = new ImageProcessor[lines.length];
		for(int i = 0; i<lines.length;i++) {
			linesIP[i] = lines[i].getProcessor();
		}
		
		int w = oIP.getWidth();
		int h = oIP.getHeight();
		//create blank 8-bit grayscale images in the correct size
		ImagePlus resImage = IJ.createImage("ex_5_3_Sub_"+designation+".png", "8-Bit",w,h,1);
		ImageProcessor resIp = resImage.getProcessor();

		int[] cOrig = new int[3];
		int[] cLines = new int[3];
		int newVal=0;
		int cCombine =0;
		for(int u = 0; u<w;u++)
		{
			for(int v = 0; v<h;v++)
			 {
				cCombine = 0; 
				for(int i = 0; i<lines.length;i++) {
					cCombine += (linesIP[i].getPixel(u,v,cLines))[0];
				}
				cCombine = 255-cCombine;
				cOrig = oIP.getPixel(u,v,cOrig);
				newVal = Math.abs(cOrig[0]-cCombine);
				resIp.putPixel(u,v,newVal);
			}
		}
		//display images	
		resImage.show();
		resImage.draw();
		if(save) {
			IJ.save(resImage, resImage.getTitle());
		}
		return resImage;
	}

}
