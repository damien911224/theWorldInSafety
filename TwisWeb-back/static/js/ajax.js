
$(function(){
	$('#list_form_button').click(function(){
		$.ajax({
			type: "POST",
			url: "search_disease/",
			data: {
				'search_list': $('#clickResultView').val(),
				'csrfmiddlewaretoken': $("input[name=csrfmiddlewaretoken]").val()
			},
			success: SendSuccess,
			dataType: 'html'
		});
	});
});

function SendSuccess(data, textStatus, jqXHR){
	$('#search-prescription-results').html(data);
}

