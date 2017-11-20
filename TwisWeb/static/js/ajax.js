
$(function(){
	$('#search_prescription').keyup(function() {
		$.ajax({
			type: "POST",
			url: "search_prescription/",
			data: {
				'search_text': $('#search_prescription').val(),
				'csrfmiddlewaretoken': $("input[name=csrfmiddlewaretoken]").val()
			},
			success: prescriptionSearchSuccess,
			dataType: 'html'
		});
	});

    $('#search-prescription-results').on("click", "tr", function(){
    	var tr = $(this);
		var td = tr.children();
		$('#clickResultView').val($('#clickResultView').val() + td.eq(1).text());
    });

	$('#list_form_button').click(function(){
//	    $("#loading").html("<img id='loading-image' src='../../static/img/loading.gif' alt='Loading...' />");
//	    $("#loading").css(" width: 100%;height: 100%;top: 0px;left: 0px;position: fixed;display: block;opacity: 0.7;background-color: #fff;z-index: 99;text-align: center; ");
//	    $("#loading-image").css("position: absolute;top: 50%;left: 50%;z-index: 100; ");
//	     //<img id="loading-image" src="../../../static/img/loading.gif" alt="Loading..." />
		$.ajax({
			type: "POST",
			url: "search_disease/",
			data: {
				'search_list': $('#clickResultView').val(),
				'csrfmiddlewaretoken': $("input[name=csrfmiddlewaretoken]").val()
			},
			success: diseaseSearchSuccess,
			dataType: 'html'
		});
	});

//	$('#search-disease-results').on({
//	    click : function(){
//			var tr = $(this);
//			var td2 = tr.children();//선택한 상병값
//			td2.eq(0).css("background-color", "#abd9e9");
//			var pre_td = td2.eq(1);
//
//			//여기서 ajax로 관련처방리스트 불러서 아래의 pre_td.text()안에 넣으면댐
//			$.ajax({
//				type: "POST",
//				url: "search_connection/",
//				data: {
//					'connection_list': pre_td.text(),
//					'csrfmiddlewaretoken': $("input[name=csrfmiddlewaretoken]").val()
//				},
//				success: connectionSearchSuccess,
//				dataType: 'html'
//			})
//			
//			//var disease_name = td.text();
//			//var connection = "{{ connection }}";
//			
//			//console.log(connection);
//			pre_td.text(td2.text());
//
//		}/*,
//	    mouseleave: function(){
//	        var tr = $(this);
//            var td = tr.children();//선택한 상병값
//            var pre_td = td.eq(1);
//            td.eq(0).css("background-color", "#ffffff");
//            pre_td.text("")
//	    }*/
//
//	}, "tr");


});

function prescriptionSearchSuccess(data, textStatus, jqXHR){
	$('#search-prescription-results').html(data);
}

function diseaseSearchSuccess(data, textStatus, jqXHR){
    $('#loading').attr('style', 'visibility:hidden');
	$('#resultDiv').html(data);
    //$('#testView').html(data)

}

function connectionSearchSuccess(data, textStatus, jqXHR){
	$('#search--results').html(data);
}
