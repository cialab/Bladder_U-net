function [names,masks] = getMasks(img_path,xml_path, mask_name)

% Read image, metadata, and get width/height
infoo=imfinfo(img_path);
info=[infoo(1);infoo(3:length(infoo)-2)];
info_i=[1 linspace(3,length(infoo)-2,length(infoo)-4)];

% Get top image size for mask
xSize=info(1).Height;
ySize=info(1).Width;

% Read in XML file
xDoc = xmlread(xml_path);

% Get annotation labels
annotations = xDoc.getElementsByTagName('Annotation');

% Go through labels
masks=cell(annotations.getLength,1);
names=cell(annotations.getLength,1);
for a=0:annotations.getLength-1
    an = annotations.item(a); % dealing with one now

    % Get name of annotation
    n = an.getElementsByTagName('Attribute');
    n = n.item(0);
    name = n.getAttribute('Value');

    % Get the list of regions
    regions=an.getElementsByTagName('Region');

    for regioni = 0:regions.getLength-1
        region=regions.item(regioni);  % for each region tag

        % Get a list of all the vertexes
        verticies=region.getElementsByTagName('Vertex');
        xy{regioni+1}=zeros(verticies.getLength-1,2); % allocate space for them
        for vertexi = 0:verticies.getLength-1 % iterate through all verticies
            % Get the x value of that vertex
            x=str2double(verticies.item(vertexi).getAttribute('X'));

            % Get the y value of that vertex
            y=str2double(verticies.item(vertexi).getAttribute('Y'));
            xy{regioni+1}(vertexi+1,:)=[x,y]; % finally save them into the array
        end
    end

    % No annotations found
    if isempty(xy)
        fprintf('No annotations found\nSkipping\n\n');
        continue
    end

    regions={};
    r_i=1;
    while ~isempty(xy)
        %Collect end points
        points=zeros(length(xy)*2,2);
        for i=1:length(xy)
            points(i,1)=xy{i}(1,1);
            points(i,2)=xy{i}(1,2);
            points(i+length(xy),1)=xy{i}(length(xy{i}),1);
            points(i+length(xy),2)=xy{i}(length(xy{i}),2);
        end

        % Compute distances between each end point
        p=pdist(points,'Euclidean');
        s=squareform(p);
        s=s+diag(inf(1,size(s,1)));
        minMatrix=min(s(:));
        [row,col]=find(s==minMatrix);

        % Orient the paths of the selected endpoints in the same direction
        if col(1)<=length(xy) && row(1)<=length(xy)
            xy{row(1)}=flipud(xy{row(1)});
        end
        if col(1)>length(xy) && row(1)>length(xy)
            xy{col(1)-length(xy)}=flipud(xy{col(1)-length(xy)});
        end

        % Fix indexes
        if row(1)>length(xy)
            row(1)=row(1)-length(xy);
        end
        if col(1)>length(xy)
            col(1)=col(1)-length(xy);
        end

        % Merge the paths
        if row(1)==col(1)   % this means the region is complete
            regions{r_i}=xy{row(1)};
            r_i=r_i+1;
            xy={xy{1:col(1)-1},xy{col(1)+1:length(xy)}};
        else
            c=[xy{row(1)};xy{col(1)}];  % merging
            if row(1)<col(1)
                xy{row(1)}=c;
                xy={xy{1:col(1)-1},xy{col(1)+1:length(xy)}};
            else
                xy{col(1)}=c;
                xy={xy{1:row(1)-1},xy{row(1)+1:length(xy)}};
            end
        end
    end
    
    m=zeros(xSize,ySize,'logical');
    for i=1:length(regions)
        r=regions{i};
        m=m|poly2mask(r(:,1),r(:,2),xSize,ySize);
    end
    masks{a+1}=m;
    names{a+1}=char(name);
   
end
    types = ["Lamina Propria", "Muscularis Propria", "Mucosa", "RBC", "Cautery", "Inflammation", "Muscularis Mucosa"];
    color = 0;
    superimpose = cell(length(masks),1);
    for i=1:length(masks)
        if ~isempty(masks{i})
            if ~ismember(char(names{i}), types)
                names{i} = "Other";
                color = 9;
            else
                color = find(types==char(names{i})) + 1;
            end
             
            fprintf("WRITING");
            masks{i} = 12*uint8(masks{i});
            masks{i}(masks{i}==12) = color
            superimpose{i}=masks{i};
            %imwrite(superimpose{i},strcat(strcat(char(names{i}), num2str(i)), '.png'),'PNG');
        end
    end
    combined = superimpose{1};
    overlap = (superimpose{1}~=0);
    for i=2:length(superimpose)
        if ~isempty(superimpose{i})
            combined=combined+superimpose{i};%imfuse(combined, superimpose{i});
            overlap=overlap+(superimpose{i}~=0);
            %combined(combined==0) = superimpose{i}(superimpose{i}~=0);
            %imwrite(combined,strcat(strcat("teststst", num2str(i)), '.png'),'PNG');
        end
    end
    z=combined;
    z(overlap>1)=0;
    
    %imwrite(z,mask_name,'PNG');
	% Get the dimensions of the image.  numberOfColorBands should be = 3.
	[rows columns numberOfColorBands] = size(z)
	mask_name = extractBefore(mask_name, ".")

	%==========================================================================
	% The first way to divide an image up into blocks is by using mat2cell().
	blockSizeR = 512; % Rows in block.
	blockSizeC = 512; % Columns in block.

	% Figure out the size of each block in rows.
	% Most will be blockSizeR but there may be a remainder amount of less than that.
	wholeBlockRows = floor(rows / blockSizeR);
	blockVectorR = [blockSizeR * ones(1, wholeBlockRows), rem(rows, blockSizeR)];
	% Figure out the size of each block in columns.
	wholeBlockCols = floor(columns / blockSizeC);
	blockVectorC = [blockSizeC * ones(1, wholeBlockCols), rem(columns, blockSizeC)];

	% Create the cell array, ca. 
	% Each cell (except for the remainder cells at the end of the image)
	% in the array contains a blockSizeR by blockSizeC by 3 color array.
	% This line is where the image is actually divided up into blocks.
	if numberOfColorBands > 1
		% It's a color image.
		ca = mat2cell(z, blockVectorR, blockVectorC, numberOfColorBands);
	else
		ca = mat2cell(z, blockVectorR, blockVectorC);
	end

	% Now display all the blocks.
	plotIndex = 1;
	numPlotsR = size(ca, 1);
	numPlotsC = size(ca, 2);
	for r = 1 : numPlotsR
		for c = 1 : numPlotsC
			%fprintf('plotindex = %d,   c=%d, r=%d\n', plotIndex, c, r);
			% Specify the location for display of the image.
			%subplot(numPlotsR, numPlotsC, plotIndex);
			% Extract the numerical array out of the cell
			% just for tutorial purposes.
			rgbBlock = ca{r,c};
			%imshow(rgbBlock); % Could call imshow(ca{r,c}) if you wanted to.
			%[rowsB columnsB numberOfColorBandsB] = size(rgbBlock);
			% Make the caption the block number.
			%caption = sprintf('Block #%d of %d\n%d rows by %d columns', ...
			%	plotIndex, numPlotsR*numPlotsC, rowsB, columnsB);
			%title(caption);
			%drawnow;
			% Increment the subplot to the next location.
			imwrite(rgbBlock,strcat(mask_name,"-",num2str(c-1),"-",num2str(r-1),'.png'),'PNG');
			plotIndex = plotIndex + 1;
		end
	end
end