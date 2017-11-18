#!/bin/bash

TSN_ROOT=/home/damien/theWorldInSafety/actionRecognition
TOOLS=$TSN_ROOT/lib/caffe-action/build/tools

OLD_VERSION=3
NEW_VERSION=4
NUM_OF_SPLITS=3
SPATIAL_ITER=3500
TEMPORAL_ITER=18000

for ((SPLIT=1; SPLIT<=${NUM_OF_SPLITS}; SPLIT++))
do
    echo "TWIS Training V${NEW_VERSION} Split ${SPLIT} Start"

    if [ $SPLIT -eq 1 ]
    then	    
	    SPATIAL_WEIGHTS=$TSN_ROOT/models/twis_caffemodels/v$OLD_VERSION/twis_spatial_net_v${OLD_VERSION}.caffemodel
	    TEMPORAL_WEIGHTS=$TSN_ROOT/models/twis_caffemodels/v$OLD_VERSION/twis_temporal_net_v${OLD_VERSION}.caffemodel
    else
	    SPATIAL_WEIGHTS=$TSN_ROOT/models/twis_caffemodels/v$NEW_VERSION/split$(($SPLIT - 1))/twis_spatial_net_v${NEW_VERSION}_iter_${SPATIAL_ITER}.caffemodel
	    TEMPORAL_WEIGHTS=$TSN_ROOT/models/twis_caffemodels/v$NEW_VERSION/split$(($SPLIT - 1))/twis_temporal_net_v${NEW_VERSION}_iter_${TEMPORAL_ITER}.caffemodel
	fi

    SPATIAL_SOLVER=$TSN_ROOT/models/twis/tsn_bn_inception_rgb_solver_split_$SPLIT.prototxt
	TEMPORAL_SOLVER=$TSN_ROOT/models/twis/tsn_bn_inception_flow_solver_split_$SPLIT.prototxt

	$TOOLS/caffe train \
      --solver $SPATIAL_SOLVER \
	  --weights $SPATIAL_WEIGHTS

    $TOOLS/caffe train \
      --solver $TEMPORAL_SOLVER \
      --weights $TEMPORAL_WEIGHTS

	if [ $SPLIT -eq $NUM_OF_SPLITS ]
	then	    
        cp $TSN_ROOT/models/twis_caffemodels/v${NEW_VERSION}/split${SPLIT}/twis_spatial_net_v${NEW_VERSION}_iter_${SPATIAL_ITER}.caffemodel \
           $TSN_ROOT/models/twis_caffemodels/v${NEW_VERISON}/twis_spatial_net_v${NEW_VERSION}.caffemodel
        cp $TSN_ROOT/models/twis_caffemodels/v${NEW_VERSION}/split${SPLIT}/twis_temporal_net_v${NEW_VERSION}_iter_${TEMPORAL_ITER}.caffemodel \
           $TSN_ROOT/models/twis_caffemodels/v${NEW_VERSION}/twis_temporal_net_v${NEW_VERSION}.caffemodel
    fi
done

echo "TWIS Training V${NEW_VERSION} ALL DONE"

